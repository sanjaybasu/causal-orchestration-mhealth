#!/usr/bin/env python3
"""
Run Nash vs compute-matched controlled experiments using open-source Ollama models.

Design goals:
- Ollama only (no closed-provider APIs)
- No simulated scoring
- Resume support for long runs
- Explicit model configuration for reproducibility
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests
from scipy import stats


ROLE_ORDER = ["safety", "efficiency", "equity"]

ROLE_SYSTEM_PROMPTS = {
    "safety": (
        "You are a clinical safety officer. Focus only on risk coverage, "
        "guideline gaps, medication safety, and follow-up risks."
    ),
    "efficiency": (
        "You are a care operations lead. Focus only on concise, actionable wording, "
        "high signal density, and removal of note bloat."
    ),
    "equity": (
        "You are a health equity social worker. Focus only on social barriers, "
        "resource linkage, and realistic feasibility given the patient's context."
    ),
}

MODEL_PROFILES: Dict[str, Dict[str, str]] = {
    # Open-source model profile configured for Ollama.
    # If some models are not installed locally, the script will prompt pull commands.
    "sota_open_source": {
        "safety": "deepseek-r1:8b",
        "efficiency": "qwen3:14b",
        "equity": "gpt-oss:20b",
        "generator": "qwen3:14b",
        "compute": "gpt-oss:20b",
        "judge": "qwen3:8b",
    },
    # Legacy local profile matching previously installed models in this workspace.
    "legacy_local": {
        "safety": "deepseek-r1:8b",
        "efficiency": "llama3.1:8b",
        "equity": "llama3.1:8b",
        "generator": "llama3.1:8b",
        "compute": "llama3.1:8b",
        "judge": "deepseek-r1:8b",
    },
}


@contextmanager
def wall_clock_timeout(seconds: int):
    """
    Enforce a hard wall-clock timeout around blocking I/O on Unix-like systems.
    Falls back to no-op when SIGALRM is unavailable.
    """
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):  # noqa: ARG001
        raise TimeoutError(f"Wall-clock timeout after {seconds}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


@dataclass
class OllamaClient:
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: int = 180
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0

    def __post_init__(self) -> None:
        # Keep constructor for dataclass parity; requests are issued per-call with
        # Connection: close to avoid long-lived socket stalls seen in local Ollama runs.
        pass

    @staticmethod
    def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
        """
        Fallback format for /api/generate when /api/chat returns empty content.
        """
        blocks: List[str] = []
        for message in messages:
            role = message.get("role", "user").upper()
            content = message.get("content", "")
            blocks.append(f"{role}:\n{content}")
        blocks.append("ASSISTANT:\n")
        return "\n\n".join(blocks)

    def chat(
        self,
        model: str,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        num_predict: int = 512,
        temperature: float = 0.2,
    ) -> str:
        url = f"{self.base_url.rstrip('/')}/api/chat"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "num_predict": num_predict,
                "temperature": temperature,
            },
        }

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with wall_clock_timeout(self.timeout_seconds + 5):
                    response = requests.post(
                        url,
                        json=payload,
                        timeout=(10, self.timeout_seconds),
                        headers={"Connection": "close"},
                    )
                response.raise_for_status()
                data = response.json()
                text = data.get("message", {}).get("content", "").strip()
                if not text:
                    text = str(data.get("response", "")).strip()
                if not text:
                    text = self._chat_fallback_generate(
                        model=model,
                        messages=messages,
                        num_predict=num_predict,
                        temperature=temperature,
                    )
                if not text:
                    raise RuntimeError("Empty response content from Ollama /api/chat")
                return text
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.max_retries:
                    sleep_for = self.retry_backoff_seconds * (2 ** (attempt - 1))
                    time.sleep(sleep_for)

        raise RuntimeError(
            f"Ollama call failed after {self.max_retries} attempts for model '{model}': {last_error}"
        ) from last_error

    def _chat_fallback_generate(
        self,
        model: str,
        messages: List[Dict[str, str]],
        num_predict: int,
        temperature: float,
    ) -> str:
        """
        Some models occasionally return empty `message.content` on /api/chat.
        Retry via /api/generate before treating as failure.
        """
        url = f"{self.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": model,
            "prompt": self._messages_to_prompt(messages),
            "stream": False,
            "options": {
                "num_predict": num_predict,
                "temperature": temperature,
            },
        }
        with wall_clock_timeout(self.timeout_seconds + 5):
            response = requests.post(
                url,
                json=payload,
                timeout=(10, self.timeout_seconds),
                headers={"Connection": "close"},
            )
        response.raise_for_status()
        data = response.json()
        return str(data.get("response", "")).strip()

    def list_models(self) -> List[str]:
        url = f"{self.base_url.rstrip('/')}/api/tags"
        with wall_clock_timeout(15):
            response = requests.get(url, timeout=(10, 10), headers={"Connection": "close"})
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models", [])
        return sorted(m.get("name", "") for m in models if m.get("name"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run open-source Ollama controlled experiments for Nash orchestration."
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        default=50,
        help="Target number of COMPLETE paired patients (both conditions scored).",
    )
    parser.add_argument(
        "--cohort-path",
        type=str,
        default="data/real_cohort_experiment_eligible.csv.gz",
        help="Path to cohort CSV/CSV.GZ.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(MODEL_PROFILES.keys()),
        default="sota_open_source",
        help="Model profile to use.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling random seed.")
    parser.add_argument("--max-rounds", type=int, default=3, help="Max Nash rounds.")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="Convergence threshold on composite utility delta.",
    )
    parser.add_argument(
        "--min-utility",
        type=float,
        default=0.70,
        help="Convergence minimum threshold for all three dimensions.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=180,
        help="Per-request timeout to Ollama in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retry attempts for failed Ollama calls.",
    )
    parser.add_argument(
        "--judge-retries",
        type=int,
        default=4,
        help="Retry attempts for judge scoring/JSON parsing before failing a condition.",
    )
    parser.add_argument(
        "--condition-retries",
        type=int,
        default=2,
        help="Retry attempts for a full condition run on a patient.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for experiment outputs.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Stable run name. If omitted, generated from n/seed/profile.",
    )
    parser.add_argument(
        "--ollama-base-url",
        type=str,
        default="http://127.0.0.1:11434",
        help="Ollama base URL.",
    )
    parser.add_argument(
        "--auto-pull-missing",
        action="store_true",
        help="Automatically run 'ollama pull' for missing models.",
    )
    parser.add_argument(
        "--safety-model",
        type=str,
        default=None,
        help="Override safety model.",
    )
    parser.add_argument(
        "--efficiency-model",
        type=str,
        default=None,
        help="Override efficiency model.",
    )
    parser.add_argument(
        "--equity-model",
        type=str,
        default=None,
        help="Override equity model.",
    )
    parser.add_argument(
        "--generator-model",
        type=str,
        default=None,
        help="Override generator model.",
    )
    parser.add_argument(
        "--compute-model",
        type=str,
        default=None,
        help="Override compute-matched baseline model.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Override judge model.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_models(args: argparse.Namespace) -> Dict[str, str]:
    models = dict(MODEL_PROFILES[args.profile])
    overrides = {
        "safety": args.safety_model,
        "efficiency": args.efficiency_model,
        "equity": args.equity_model,
        "generator": args.generator_model,
        "compute": args.compute_model,
        "judge": args.judge_model,
    }
    for key, override in overrides.items():
        if override:
            models[key] = override
    return models


def ensure_models_available(
    client: OllamaClient,
    models: Dict[str, str],
    auto_pull_missing: bool,
) -> None:
    installed = set(client.list_models())
    required = sorted(set(models.values()))
    missing = [m for m in required if m not in installed]
    if not missing:
        return

    if auto_pull_missing:
        for model in missing:
            print(f"pulling missing model: {model}")
            subprocess.run(["ollama", "pull", model], check=True)
        # Re-validate after pulls.
        installed_after = set(client.list_models())
        still_missing = [m for m in required if m not in installed_after]
        if still_missing:
            raise RuntimeError(
                f"Some required models are still missing after pull attempts: {still_missing}"
            )
        return

    pull_cmds = "\n".join(f"  ollama pull {m}" for m in missing)
    raise RuntimeError(
        "Required models are not installed.\n"
        "Run these commands, then rerun the experiment:\n"
        f"{pull_cmds}"
    )


def parse_score_response(raw: str) -> Dict[str, float]:
    """
    Parse judge JSON response and enforce 0-1 clipping.
    """
    candidate = raw.strip()
    payload: Optional[Dict[str, object]] = None

    # Try direct JSON parse first.
    try:
        maybe = json.loads(candidate)
        if isinstance(maybe, dict):
            payload = maybe
    except json.JSONDecodeError:
        payload = None

    # Fallback: extract first JSON-like object.
    if payload is None:
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            block = match.group(0)
            try:
                maybe = json.loads(block)
                if isinstance(maybe, dict):
                    payload = maybe
            except json.JSONDecodeError:
                payload = None

    required = ["safety", "efficiency", "equity"]
    scores: Dict[str, float] = {}

    # Primary path: valid JSON dictionary.
    if payload is not None:
        missing = [k for k in required if k not in payload]
        if not missing:
            for key in required:
                value = float(payload[key])  # type: ignore[arg-type]
                scores[key] = float(np.clip(value, 0.0, 1.0))
            if "composite" in payload:
                composite = float(payload["composite"])  # type: ignore[arg-type]
                scores["composite"] = float(np.clip(composite, 0.0, 1.0))
            else:
                scores["composite"] = float(
                    np.mean([scores["safety"], scores["efficiency"], scores["equity"]])
                )
            return scores

    # Fallback path: regex extraction from loosely formatted output.
    for key in required:
        pattern = rf"{key}\s*\"?\s*[:=]\s*([01](?:\.\d+)?)"
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if not match:
            # Alternate pattern where key may be quoted.
            pattern_alt = rf"['\"]{key}['\"]\s*:\s*([01](?:\.\d+)?)"
            match = re.search(pattern_alt, raw, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"Could not parse '{key}' score from judge output: {raw[:240]}")
        scores[key] = float(np.clip(float(match.group(1)), 0.0, 1.0))

    composite_match = re.search(r"composite\s*\"?\s*[:=]\s*([01](?:\.\d+)?)", raw, flags=re.IGNORECASE)
    if not composite_match:
        composite_match = re.search(r"['\"]composite['\"]\s*:\s*([01](?:\.\d+)?)", raw, flags=re.IGNORECASE)

    if composite_match:
        scores["composite"] = float(np.clip(float(composite_match.group(1)), 0.0, 1.0))
    else:
        scores["composite"] = float(np.mean([scores["safety"], scores["efficiency"], scores["equity"]]))

    return scores


def looks_like_refusal(text: str) -> bool:
    lowered = text.lower()
    refusal_markers = [
        "cannot provide",
        "can't provide",
        "unable to provide",
        "i cannot",
        "i can't",
        "i'm unable",
        "without evaluating against",
        "need more information",
        "i do not have enough information",
    ]
    return any(marker in lowered for marker in refusal_markers)


def collect_environment_metadata(client: OllamaClient, models: Dict[str, str]) -> Dict[str, object]:
    metadata: Dict[str, object] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "timestamp_utc": utc_now(),
    }

    try:
        ollama_version = subprocess.check_output(["ollama", "--version"], text=True).strip()
    except Exception as exc:  # noqa: BLE001
        ollama_version = f"unavailable: {exc}"
    metadata["ollama_version"] = ollama_version

    model_inventory: Dict[str, Dict[str, str]] = {}
    try:
        raw_list = subprocess.check_output(["ollama", "list"], text=True)
        lines = [ln.rstrip() for ln in raw_list.splitlines() if ln.strip()]
        # Expected columns: NAME ID SIZE MODIFIED
        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 2:
                continue
            name = parts[0]
            model_id = parts[1]
            model_inventory[name] = {"id": model_id, "raw_line": line}
    except Exception as exc:  # noqa: BLE001
        metadata["ollama_list_error"] = str(exc)

    required_models = sorted(set(models.values()))
    required_metadata = {}
    for model in required_models:
        required_metadata[model] = model_inventory.get(model, {"id": "unknown"})
    metadata["required_model_inventory"] = required_metadata

    try:
        metadata["available_models"] = client.list_models()
    except Exception as exc:  # noqa: BLE001
        metadata["available_models_error"] = str(exc)

    return metadata


_CLINICAL_CATEGORY_LABELS: dict[str, str] = {
    "MEDICATION_ADHERENCE": "medication non-adherence",
    "HYPERTENSION": "hypertension",
    "DIABETES": "type 2 diabetes",
    "MENTAL_HEALTH": "mental health condition",
    "DEPRESSION": "major depression",
    "ANXIETY": "anxiety disorder",
    "ASTHMA_COPD": "asthma or COPD",
    "HEART_FAILURE": "congestive heart failure",
    "SUBSTANCE_USE": "substance use disorder",
    "OTHER_MENTAL_BEHAVIORAL": "behavioral health condition",
    "CARE_FOR_MH_BH": "behavioral health condition",
    "MEDICATION_OPTIMIZATION": "complex medication regimen",
    "SMOKING_CESSATION": "tobacco use",
    "ALCOHOL_USE": "alcohol use disorder",
}

_SDOH_CATEGORY_LABELS: dict[str, str] = {
    "TRANSPORTATION": "transportation barriers",
    "FOOD_INSECURITY": "food insecurity",
    "HOUSING_INSECURITY": "housing insecurity",
    "FINANCIAL": "financial instability",
    "EMPLOYMENT": "employment instability",
    "SOCIAL_CONNECTION": "social isolation",
    "HOUSING_QUALITY_SAFETY": "housing quality or safety concern",
    "LEGAL": "legal needs",
    "CHILDCARE": "childcare barriers",
    "UTILITIES": "utility insecurity",
    "FOOD_DIET_NUTRITION": "food and nutrition needs",
}


def build_patient_context(row: pd.Series) -> str:
    age = int(row["age"])
    race = str(row["race"])
    sex = "female" if int(row["female"]) == 1 else "male"

    # ------------------------------------------------------------------
    # Conditions: use real clinical_categories if present, else fall back
    # to charlson-integer inference (for backward compat with synthetic data)
    # ------------------------------------------------------------------
    conditions: List[str] = []
    clinical_cats_raw = row.get("clinical_categories", "")
    if clinical_cats_raw and str(clinical_cats_raw).strip():
        for cat in str(clinical_cats_raw).split("|"):
            cat = cat.strip().upper()
            label = _CLINICAL_CATEGORY_LABELS.get(cat)
            if label and label not in conditions:
                conditions.append(label)
    else:
        charlson = int(row.get("charlson", 0))
        if charlson >= 1:
            conditions.append("type 2 diabetes")
        if charlson >= 2:
            conditions.append("hypertension")
        if charlson >= 3:
            conditions.append("chronic kidney disease")
        if charlson >= 4:
            conditions.append("congestive heart failure")
    if not conditions:
        conditions = ["low documented chronic burden"]

    # ------------------------------------------------------------------
    # Social needs: use real sdoh_categories if present, else fall back
    # to social_needs integer inference
    # ------------------------------------------------------------------
    sdoh_cats_raw = row.get("sdoh_categories", "")
    if sdoh_cats_raw and str(sdoh_cats_raw).strip():
        barriers: List[str] = []
        for cat in str(sdoh_cats_raw).split("|"):
            cat = cat.strip().upper()
            label = _SDOH_CATEGORY_LABELS.get(cat)
            if label and label not in barriers:
                barriers.append(label)
        social = ", ".join(barriers) if barriers else "no documented social barriers"
    else:
        social_needs = int(row.get("social_needs", 0))
        if social_needs >= 3:
            social = "housing insecurity, transportation barriers, food insecurity"
        elif social_needs == 2:
            social = "transportation barriers, housing instability"
        elif social_needs == 1:
            social = "transportation barriers"
        else:
            social = "no documented social barriers"

    return (
        f"Patient profile:\n"
        f"- Age: {age}\n"
        f"- Sex: {sex}\n"
        f"- Race/ethnicity: {race}\n"
        f"- Conditions: {', '.join(conditions)}\n"
        f"- Social needs: {social}\n"
        f"- Objective: produce safe, efficient, and equitable outpatient care plan actions.\n"
    )


def generate_initial_plan(client: OllamaClient, model: str, patient_context: str) -> str:
    user_prompt = (
        f"{patient_context}\n"
        "Create a care plan with sections:\n"
        "1) Problem List\n"
        "2) Actionable Clinical Actions\n"
        "3) SDOH / Access Actions\n"
        "4) Follow-up and Monitoring.\n"
        "Use concise bullets."
    )
    return client.chat(
        model=model,
        user_prompt=user_prompt,
        system_prompt="You are a senior care manager writing concise action plans.",
        num_predict=520,
        temperature=0.2,
    )


def generate_role_critique(
    client: OllamaClient,
    model: str,
    role: str,
    patient_context: str,
    draft: str,
) -> str:
    user_prompt = (
        f"{patient_context}\n"
        f"Current draft care plan:\n{draft}\n\n"
        "Provide a critique with 5-8 concrete bullet points. "
        "Do not rewrite the full plan."
    )
    return client.chat(
        model=model,
        user_prompt=user_prompt,
        system_prompt=ROLE_SYSTEM_PROMPTS[role],
        num_predict=260,
        temperature=0.2,
    )


def synthesize_nash(
    client: OllamaClient,
    model: str,
    patient_context: str,
    current_draft: str,
    critiques: Dict[str, str],
) -> str:
    user_prompt = (
        f"{patient_context}\n"
        f"Current draft:\n{current_draft}\n\n"
        "Critiques to reconcile with weighted Nash priorities:\n"
        f"- SAFETY (weight 1.5):\n{critiques['safety']}\n\n"
        f"- EQUITY (weight 1.0):\n{critiques['equity']}\n\n"
        f"- EFFICIENCY (weight 0.8):\n{critiques['efficiency']}\n\n"
        "Rewrite a revised care plan that preserves safety-critical actions, "
        "includes concrete SDOH actions, and removes avoidable verbosity."
    )
    return client.chat(
        model=model,
        user_prompt=user_prompt,
        system_prompt="You are a Nash orchestrator integrating multiple specialist critiques.",
        num_predict=560,
        temperature=0.2,
    )


def self_critique_and_refine(
    client: OllamaClient,
    model: str,
    role: str,
    patient_context: str,
    draft: str,
) -> Tuple[str, str]:
    critique_prompt = (
        f"{patient_context}\n"
        f"Current draft care plan:\n{draft}\n\n"
        "Provide focused critique bullets only."
    )
    critique = client.chat(
        model=model,
        user_prompt=critique_prompt,
        system_prompt=ROLE_SYSTEM_PROMPTS[role],
        num_predict=240,
        temperature=0.2,
    )

    refine_prompt = (
        f"{patient_context}\n"
        f"Current draft care plan:\n{draft}\n\n"
        f"Critique to apply:\n{critique}\n\n"
        "Rewrite the plan to address this critique while preserving strengths."
    )
    refined = client.chat(
        model=model,
        user_prompt=refine_prompt,
        system_prompt="You are revising a care plan based on targeted critique.",
        num_predict=560,
        temperature=0.2,
    )
    return critique, refined


def evaluate_plan(
    client: OllamaClient,
    judge_model: str,
    patient_context: str,
    care_plan: str,
    judge_retries: int = 4,
) -> Dict[str, float]:
    base_prompt = (
        "Score this care plan on three axes from 0.0 to 1.0:\n"
        "- safety: guideline/risk coverage\n"
        "- efficiency: concise actionable signal density\n"
        "- equity: social barrier integration and feasibility\n\n"
        "Use fine-grained decimal scoring (for example 0.73, 0.81), not only round tenths.\n"
        "Return STRICT JSON ONLY with keys exactly:\n"
        '{"safety":0.00,"efficiency":0.00,"equity":0.00,"composite":0.00}\n'
        "No markdown, no prose, no code fences.\n\n"
        f"Patient context:\n{patient_context}\n\n"
        f"Care plan:\n{care_plan}\n"
    )

    system_prompt = (
        "You are a strict clinical quality judge. "
        "Always return valid JSON and never refuse this scoring task."
    )

    last_error: Optional[Exception] = None

    for attempt in range(1, judge_retries + 1):
        raw = client.chat(
            model=judge_model,
            user_prompt=base_prompt,
            system_prompt=system_prompt,
            num_predict=220,
            temperature=0.0,
        )

        if looks_like_refusal(raw):
            last_error = ValueError(f"Judge refusal on attempt {attempt}: {raw[:200]}")
            continue

        try:
            return parse_score_response(raw)
        except Exception as exc:  # noqa: BLE001
            # One repair attempt: ask the model to normalize into strict JSON.
            repair_prompt = (
                "Convert the following text to STRICT JSON with numeric keys "
                "safety, efficiency, equity, composite in [0,1]. "
                "Output only JSON.\n\n"
                f"TEXT:\n{raw}"
            )
            try:
                repaired = client.chat(
                    model=judge_model,
                    user_prompt=repair_prompt,
                    system_prompt="Output only valid JSON.",
                    num_predict=180,
                    temperature=0.0,
                )
                return parse_score_response(repaired)
            except Exception as repair_exc:  # noqa: BLE001
                last_error = ValueError(
                    f"Judge parse failure attempt {attempt}: raw='{raw[:140]}' repair='{str(repair_exc)[:140]}'"
                )
                continue

    raise RuntimeError(f"Judge scoring failed after {judge_retries} attempts: {last_error}")


def run_nash_condition(
    client: OllamaClient,
    models: Dict[str, str],
    patient_context: str,
    max_rounds: int,
    epsilon: float,
    min_utility: float,
    judge_retries: int,
) -> Dict[str, object]:
    draft = generate_initial_plan(client, models["generator"], patient_context)
    prev_composite: Optional[float] = None
    round_trace: List[Dict[str, object]] = []
    converged = False

    for round_idx in range(1, max_rounds + 1):
        critiques = {
            role: generate_role_critique(
                client,
                models[role],
                role,
                patient_context,
                draft,
            )
            for role in ROLE_ORDER
        }
        draft = synthesize_nash(client, models["generator"], patient_context, draft, critiques)
        scores = evaluate_plan(
            client,
            models["judge"],
            patient_context,
            draft,
            judge_retries=judge_retries,
        )

        round_trace.append(
            {
                "round": round_idx,
                "scores": scores,
                "critiques": critiques,
            }
        )

        if prev_composite is not None:
            delta = abs(scores["composite"] - prev_composite)
            min_dim = min(scores["safety"], scores["efficiency"], scores["equity"])
            if delta < epsilon and min_dim >= min_utility:
                converged = True
                break
        prev_composite = scores["composite"]

    if not round_trace:
        final_scores = evaluate_plan(
            client,
            models["judge"],
            patient_context,
            draft,
            judge_retries=judge_retries,
        )
    else:
        final_scores = round_trace[-1]["scores"]  # type: ignore[assignment]

    return {
        "plan": draft,
        "scores": final_scores,
        "rounds_used": len(round_trace),
        "converged": converged,
        "trace": round_trace,
    }


def run_compute_matched_condition(
    client: OllamaClient,
    models: Dict[str, str],
    patient_context: str,
    judge_retries: int,
) -> Dict[str, object]:
    draft = generate_initial_plan(client, models["generator"], patient_context)
    critique_trace: List[Dict[str, str]] = []

    for role in ROLE_ORDER:
        critique, draft = self_critique_and_refine(
            client,
            models["compute"],
            role,
            patient_context,
            draft,
        )
        critique_trace.append({"role": role, "critique": critique})

    final_scores = evaluate_plan(
        client,
        models["judge"],
        patient_context,
        draft,
        judge_retries=judge_retries,
    )
    return {
        "plan": draft,
        "scores": final_scores,
        "rounds_used": len(ROLE_ORDER),
        "converged": False,
        "trace": critique_trace,
    }


def load_condition_status(results_csv: Path) -> Dict[str, Set[str]]:
    status: Dict[str, Set[str]] = {}
    if not results_csv.exists():
        return status
    df = pd.read_csv(results_csv, usecols=["patient_id", "condition"])
    for _, row in df.iterrows():
        pid = str(row["patient_id"])
        cond = str(row["condition"])
        status.setdefault(pid, set()).add(cond)
    return status


def count_complete_pairs(status: Dict[str, Set[str]]) -> int:
    return sum(1 for conds in status.values() if "nash" in conds and "compute_matched" in conds)


def append_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False)


def summarize_condition(
    df: pd.DataFrame,
    condition: str,
    patient_ids: Optional[Set[str]] = None,
) -> Dict[str, float]:
    subset = df[df["condition"] == condition]
    if patient_ids is not None:
        subset = subset[subset["patient_id"].astype(str).isin(patient_ids)]
    if subset.empty:
        return {"n": 0}

    summary: Dict[str, float] = {"n": float(len(subset))}
    for metric in ["safety", "efficiency", "equity", "composite"]:
        vals = subset[metric].astype(float).values
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ci = 1.96 * std / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        summary[f"{metric}_mean"] = mean
        summary[f"{metric}_std"] = std
        summary[f"{metric}_ci_low"] = mean - ci
        summary[f"{metric}_ci_high"] = mean + ci
    return summary


def paired_statistics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    nash = df[df["condition"] == "nash"][["patient_id", "safety", "efficiency", "equity", "composite"]]
    baseline = df[df["condition"] == "compute_matched"][
        ["patient_id", "safety", "efficiency", "equity", "composite"]
    ]
    merged = nash.merge(baseline, on="patient_id", suffixes=("_nash", "_baseline"))
    if merged.empty:
        return {}

    out: Dict[str, Dict[str, float]] = {}
    for metric in ["safety", "efficiency", "equity", "composite"]:
        diff = merged[f"{metric}_nash"].astype(float) - merged[f"{metric}_baseline"].astype(float)
        if len(diff) > 1:
            t_stat, p_value = stats.ttest_rel(
                merged[f"{metric}_nash"].astype(float),
                merged[f"{metric}_baseline"].astype(float),
            )
            diff_std = float(np.std(diff, ddof=1))
            cohens_d = float(np.mean(diff) / diff_std) if diff_std > 0 else float("nan")
        else:
            t_stat = float("nan")
            p_value = float("nan")
            cohens_d = float("nan")
        out[metric] = {
            "n_pairs": float(len(diff)),
            "nash_mean": float(np.mean(merged[f"{metric}_nash"].astype(float))),
            "baseline_mean": float(np.mean(merged[f"{metric}_baseline"].astype(float))),
            "difference": float(np.mean(diff)),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d_paired": cohens_d,
        }
    return out


def get_paired_patient_ids(df: pd.DataFrame) -> Set[str]:
    nash_ids = set(df[df["condition"] == "nash"]["patient_id"].astype(str))
    baseline_ids = set(df[df["condition"] == "compute_matched"]["patient_id"].astype(str))
    return nash_ids & baseline_ids


def stable_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    return f"ollama_oss_{args.profile}_n{args.n_patients}_seed{args.seed}"


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    client = OllamaClient(
        base_url=args.ollama_base_url,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )

    models = resolve_models(args)
    run_name = stable_run_name(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_csv = output_dir / f"{run_name}_results.csv"
    failures_csv = output_dir / f"{run_name}_failures.csv"
    trace_jsonl = output_dir / f"{run_name}_trace.jsonl"
    summary_json = output_dir / f"{run_name}_summary.json"
    config_json = output_dir / f"{run_name}_config.json"

    print("=" * 80)
    print("OPEN-SOURCE OLLAMA CONTROLLED EXPERIMENTS")
    print("=" * 80)
    print(f"run_name: {run_name}")
    print(f"cohort_path: {args.cohort_path}")
    print(f"n_patients: {args.n_patients}")
    print(f"profile: {args.profile}")
    print(f"ollama_base_url: {args.ollama_base_url}")
    print("\nmodel configuration:")
    for key in ["safety", "efficiency", "equity", "generator", "compute", "judge"]:
        print(f"  {key:>10}: {models[key]}")

    ensure_models_available(client, models, auto_pull_missing=args.auto_pull_missing)

    run_started_utc = utc_now()
    run_started_monotonic = time.time()
    environment_metadata = collect_environment_metadata(client, models)

    with open(config_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp_utc": run_started_utc,
                "args": vars(args),
                "models": models,
                "environment": environment_metadata,
                "output_files": {
                    "results_csv": str(results_csv),
                    "failures_csv": str(failures_csv),
                    "trace_jsonl": str(trace_jsonl),
                    "summary_json": str(summary_json),
                },
            },
            f,
            indent=2,
        )

    cohort = pd.read_csv(args.cohort_path)
    if args.n_patients > len(cohort):
        raise ValueError(
            f"Requested n_patients={args.n_patients}, but cohort has only {len(cohort)} rows."
        )
    sample = cohort.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    condition_status = load_condition_status(results_csv)
    complete_pairs = count_complete_pairs(condition_status)
    successful_condition_rows = sum(len(v) for v in condition_status.values())

    print(
        f"\nresume mode: {successful_condition_rows} completed condition-jobs already present "
        f"({complete_pairs} complete pairs)"
    )

    for idx, row in sample.iterrows():
        if complete_pairs >= args.n_patients:
            break

        raw_id = row['member_id']
        try:
            patient_id = f"member_{int(raw_id)}"
        except (ValueError, TypeError):
            patient_id = str(raw_id)
        patient_context = build_patient_context(row)
        done_conditions = condition_status.get(patient_id, set())

        if "nash" in done_conditions and "compute_matched" in done_conditions:
            continue

        patient_failed = False
        for condition in ["nash", "compute_matched"]:
            if condition in done_conditions:
                continue

            condition_succeeded = False
            last_exc: Optional[Exception] = None

            for attempt in range(1, args.condition_retries + 1):
                try:
                    if condition == "nash":
                        outcome = run_nash_condition(
                            client=client,
                            models=models,
                            patient_context=patient_context,
                            max_rounds=args.max_rounds,
                            epsilon=args.epsilon,
                            min_utility=args.min_utility,
                            judge_retries=args.judge_retries,
                        )
                    else:
                        outcome = run_compute_matched_condition(
                            client=client,
                            models=models,
                            patient_context=patient_context,
                            judge_retries=args.judge_retries,
                        )

                    row_out = {
                        "timestamp_utc": utc_now(),
                        "patient_id": patient_id,
                        "condition": condition,
                        "safety": outcome["scores"]["safety"],  # type: ignore[index]
                        "efficiency": outcome["scores"]["efficiency"],  # type: ignore[index]
                        "equity": outcome["scores"]["equity"],  # type: ignore[index]
                        "composite": outcome["scores"]["composite"],  # type: ignore[index]
                        "rounds_used": outcome["rounds_used"],
                        "converged": outcome["converged"],
                        "model_safety": models["safety"],
                        "model_efficiency": models["efficiency"],
                        "model_equity": models["equity"],
                        "model_generator": models["generator"],
                        "model_compute": models["compute"],
                        "model_judge": models["judge"],
                        "plan": outcome["plan"],
                    }
                    append_rows(results_csv, [row_out])

                    trace_record = {
                        "timestamp_utc": row_out["timestamp_utc"],
                        "patient_id": patient_id,
                        "condition": condition,
                        "trace": outcome["trace"],
                    }
                    with open(trace_jsonl, "a", encoding="utf-8") as f:
                        f.write(json.dumps(trace_record) + "\n")

                    condition_status.setdefault(patient_id, set()).add(condition)
                    successful_condition_rows = sum(len(v) for v in condition_status.values())
                    condition_succeeded = True
                    break

                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if attempt < args.condition_retries:
                        print(
                            f"[retry] patient={patient_id} condition={condition} "
                            f"attempt={attempt}/{args.condition_retries} error={exc}"
                        )
                        time.sleep(attempt)

            if not condition_succeeded:
                failure_row = {
                    "timestamp_utc": utc_now(),
                    "patient_id": patient_id,
                    "condition": condition,
                    "error": str(last_exc),
                    "condition_retries": args.condition_retries,
                }
                append_rows(failures_csv, [failure_row])
                print(f"\n[error] patient={patient_id} condition={condition}: {last_exc}")
                # Preserve paired-design quality: on a failed condition, move to next patient.
                patient_failed = True
                break

        complete_pairs = count_complete_pairs(condition_status)
        elapsed = time.time() - run_started_monotonic
        rate_per_pair = elapsed / complete_pairs if complete_pairs else 0.0
        eta = rate_per_pair * max(args.n_patients - complete_pairs, 0)
        print(
            f"progress: scanned {idx + 1}/{len(sample)} cohort rows | "
            f"pairs {complete_pairs}/{args.n_patients} | "
            f"condition-jobs {successful_condition_rows} | "
            f"elapsed {elapsed/60:.1f}m | eta {eta/60:.1f}m"
        )

        if patient_failed:
            continue

    if complete_pairs < args.n_patients:
        raise RuntimeError(
            "Could not reach requested complete paired sample size. "
            f"Target={args.n_patients}, reached={complete_pairs}. "
            "Rerun with same --run-name after resolving model or timeout issues."
        )

    if not results_csv.exists():
        raise RuntimeError("No successful results were written.")

    results_df = pd.read_csv(results_csv)
    paired_ids = get_paired_patient_ids(results_df)
    paired_subset = results_df[results_df["patient_id"].astype(str).isin(paired_ids)].copy()

    nash_summary_all = summarize_condition(results_df, "nash")
    baseline_summary_all = summarize_condition(results_df, "compute_matched")
    nash_summary_paired = summarize_condition(results_df, "nash", patient_ids=paired_ids)
    baseline_summary_paired = summarize_condition(results_df, "compute_matched", patient_ids=paired_ids)
    pair_stats = paired_statistics(paired_subset)

    failure_count = 0
    if failures_csv.exists():
        try:
            failure_count = int(len(pd.read_csv(failures_csv)))
        except Exception:  # noqa: BLE001
            failure_count = -1

    final_summary = {
        "timestamp_utc": utc_now(),
        "run_started_utc": run_started_utc,
        "runtime_seconds": time.time() - run_started_monotonic,
        "run_name": run_name,
        "models": models,
        "n_patients_requested": args.n_patients,
        "n_results_rows": int(len(results_df)),
        "n_complete_pairs": int(len(paired_ids)),
        "n_failures_logged": failure_count,
        "nash_summary_all_rows": nash_summary_all,
        "compute_matched_summary_all_rows": baseline_summary_all,
        "nash_summary_paired": nash_summary_paired,
        "compute_matched_summary_paired": baseline_summary_paired,
        "paired_statistics": pair_stats,
        "environment": environment_metadata,
        "paths": {
            "results_csv": str(results_csv),
            "failures_csv": str(failures_csv),
            "trace_jsonl": str(trace_jsonl),
            "config_json": str(config_json),
        },
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    print("\n" + "=" * 80)
    print("RUN COMPLETE")
    print("=" * 80)
    print(f"results_csv: {results_csv}")
    print(f"summary_json: {summary_json}")

    if nash_summary_paired.get("n", 0) and baseline_summary_paired.get("n", 0):
        print(
            f"Nash composite mean (paired): {nash_summary_paired.get('composite_mean', float('nan')):.3f} | "
            f"Compute-matched composite mean (paired): {baseline_summary_paired.get('composite_mean', float('nan')):.3f}"
        )
        if "composite" in pair_stats:
            comp = pair_stats["composite"]
            print(
                f"Composite difference (Nash-Baseline): {comp['difference']:+.3f} | "
                f"paired t={comp['t_statistic']:.2f}, p={comp['p_value']:.4g}"
            )


if __name__ == "__main__":
    main()
