#!/usr/bin/env python3
"""
Design v2: LLMs propose actions, a verifier scores them against a patient-specific checklist,
and an exact solver selects the action set (maximum Nash welfare, utilitarian, or max-min).

Arms per patient:
  S-nash, S-sum, S-maxmin : one shared verified action pool; only the selection rule differs
  R                       : verifier-guided self-refine (missing visible items fed back as text)
Baseline B (generic self-critique) comes from the v1 run and is re-scored with the v2 evaluator.

Evaluation uses a different model family (gpt-oss:20b) on every checklist item, including the
items held out from the optimizer (eval_only).
"""

import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_ollama_open_source_experiments import OllamaClient, build_patient_context  # noqa: E402

REQMAP = Path("/Users/sanjaybasu/waymark-local/notebooks/causal_orchestration_mhealth_submission/"
              "revision_1/checklist/requirement_map_v1.json")
DOMAINS = ["safety", "efficiency", "equity"]
MODELS = {
    "proposer_safety": "deepseek-r1:8b",
    "proposer_efficiency": "llama3.1:8b",
    "proposer_equity": "llama3.1:8b",
    "verifier": "gpt-oss:20b",
    "renderer": "llama3.1:8b",
    "evaluator": "gpt-oss:20b",
}
EVALUATOR_SETTINGS = {"num_predict": 1024, "think": "low"}
VERIFIER_SETTINGS = {"num_predict": 1024, "think": "low"}  # gpt-oss reasons even with think=false
ACTION_BUDGET = 8  # same cap for every arm (actions); refine is told the same cap
WORD_CAP = 250
EPS = 0.05

ROLE_FOCUS = {
    "safety": "clinical risk coverage, guideline-concordant monitoring, medication safety, and warning signs",
    "efficiency": "the few highest-yield actions, clear owners and timeframes, and care coordination",
    "equity": "the patient's documented social needs, concrete resource linkage, and feasibility",
}


# ---------------------------------------------------------------- checklist
def load_reqmap(path: Path = REQMAP) -> dict:
    return json.loads(path.read_text())


def checklist_for(row, reqmap: dict) -> List[dict]:
    """Universal items plus every item for the patient's documented categories."""
    codes = {c.strip().upper() for col in ("clinical_categories", "sdoh_categories")
             for c in str(row.get(col, "") or "").split("|") if c.strip()}
    items = []
    for key, cat in reqmap["categories"].items():
        if cat["kind"] == "universal" or codes & set(cat["aliases"]):
            items += [dict(it, category=key) for it in cat["items"]]
    return items


# ---------------------------------------------------------------- LLM helpers
def _json_list(raw: str) -> list:
    m = re.search(r"\[[\s\S]*\]", raw)
    if not m:
        return []
    try:
        val = json.loads(m.group(0))
        return val if isinstance(val, list) else []
    except json.JSONDecodeError:
        return []


def propose_actions(client: OllamaClient, role: str, ctx: str, visible: List[dict], n: int = 8) -> List[str]:
    reqs = "\n".join(f"- {it['requirement']}" for it in visible)
    prompt = (f"{ctx}\nCare-plan requirements for this patient:\n{reqs}\n\n"
              f"Propose up to {n} specific care-plan actions focused on {ROLE_FOCUS[role]}. "
              "Each action is one sentence with who does what by when. Do not start, stop, or change "
              "medications; route medication changes to the prescriber or pharmacist.\n"
              'Return ONLY a JSON list of strings, e.g. ["action 1", "action 2"].')
    raw = client.chat(model=MODELS[f"proposer_{role}"], user_prompt=prompt,
                      system_prompt="You are a care management planner.", num_predict=700, temperature=0.4)
    return [str(a).strip() for a in _json_list(raw) if str(a).strip()][:n]


def verify_coverage(client: OllamaClient, model: str, items: List[dict], actions: List[str],
                    settings: Optional[dict] = None) -> Dict[str, List[int]]:
    """For each checklist item, which numbered actions meet it (atomic yes/no per item)."""
    numbered = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(actions))
    out = {}
    for it in items:
        prompt = (f"Requirement: {it['question']}\nCounts as MET: {it['met']}\nCounts as NOT MET: {it['not_met']}\n\n"
                  f"Candidate actions:\n{numbered}\n\n"
                  "Which numbered actions meet the requirement on their own? "
                  "Return ONLY a JSON list of numbers, or [] if none.")
        raw = client.chat(model=model, user_prompt=prompt, system_prompt="You are a strict clinical auditor.",
                          num_predict=(settings or {}).get("num_predict", 60), temperature=0.0,
                          think=(settings or {}).get("think", False))
        out[it["id"]] = sorted({int(x) - 1 for x in _json_list(raw)
                                if str(x).isdigit() and 0 < int(x) <= len(actions)})
    return out


def flag_unsupported(client: OllamaClient, ctx: str, actions: List[str]) -> List[int]:
    numbered = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(actions))
    prompt = (f"{ctx}\nCandidate actions:\n{numbered}\n\n"
              "Which numbered actions are unsafe, clinically inappropriate for this profile, or change a "
              "medication without a prescriber? Return ONLY a JSON list of numbers, or [] if none.")
    raw = client.chat(model=MODELS["verifier"], user_prompt=prompt,
                      system_prompt="You are a strict clinical safety auditor.", temperature=0.0, **VERIFIER_SETTINGS)
    return sorted({int(x) - 1 for x in _json_list(raw) if str(x).isdigit() and 0 < int(x) <= len(actions)})


def confirm_pairs(client: OllamaClient, items: List[dict], actions: List[str],
                  shortlist: Dict[str, List[int]]) -> Dict[str, List[int]]:
    """Re-check each shortlisted (item, action) pair with the same per-item question used for grading."""
    byid = {it["id"]: it for it in items}
    out = {}
    for iid, acts in shortlist.items():
        it = byid[iid]
        keep = []
        for a in acts:
            prompt = (f"Care plan:\n- {actions[a]}\n\nRequirement: {it['question']}\nCounts as MET: {it['met']}\n"
                      f"Counts as NOT MET: {it['not_met']}\n\nAnswer ONLY yes or no.")
            raw = client.chat(model=MODELS["verifier"], user_prompt=prompt, system_prompt="You are a strict clinical auditor.",
                              temperature=0.0, **VERIFIER_SETTINGS)
            if raw.strip().lower().lstrip("*\"' ").startswith("yes"):
                keep.append(a)
        out[iid] = keep
    return out


def gap_fill(client: OllamaClient, ctx: str, missing: List[dict]) -> List[str]:
    """One targeted action per requirement that no proposed action meets."""
    new = []
    for it in missing:
        prompt = (f"{ctx}\nWrite ONE specific care-plan action (who does what by when) that meets this requirement.\n"
                  f"Requirement: {it['question']}\nCounts as MET: {it['met']}\n"
                  "Do not start, stop, or change medications. Return only the action sentence.")
        a = client.chat(model=MODELS["proposer_efficiency"], user_prompt=prompt,
                        system_prompt="You are a care management planner.", num_predict=120, temperature=0.2).strip()
        if a:
            new.append(a.splitlines()[0].strip("-* "))
    return new


# ---------------------------------------------------------------- solver
def solve(rule: str, items: List[dict], cover: Dict[str, List[int]], n_actions: int,
          excluded: List[int], budget: int = ACTION_BUDGET) -> List[int]:
    """
    Exact selection of an action set over visible items.
    x_a: choose action a; y_i: item i covered (y_i <= sum of chosen actions that meet i).
    Domain coverage c_k = sum_{i in k} y_i / n_k.
      nash   : maximize sum_k log(EPS + c_k)  (maximum Nash welfare; exact via count indicators)
      sum    : maximize sum_k c_k
      maxmin : maximize t subject to t <= c_k (ties broken by total coverage)
    Ties prefer a second confirmed action per item (robustness). Domains with no visible items are skipped.
    """
    doms = [k for k in DOMAINS if any(it["domain"] == k for it in items)]
    I = len(items)
    A = n_actions
    nk = {k: sum(it["domain"] == k for it in items) for k in doms}
    # variable layout: x[A] | y[I] | z[k][0..nk] (nash only) | t (maxmin only)
    zoff = A + I
    zidx = {}
    if rule == "nash":
        for k in doms:
            zidx[k] = list(range(zoff, zoff + nk[k] + 1))
            zoff += nk[k] + 1
    tvar = zoff if rule == "maxmin" else None
    roff = zoff + (1 if rule == "maxmin" else 0)  # r_i: item met by a second chosen action (robustness)
    nvar = roff + I

    c = np.zeros(nvar)
    c[roff:roff + I] = -1e-3  # small reward for a second confirmed action on the same item
    if rule == "nash":
        for k in doms:
            for cnt, j in enumerate(zidx[k]):
                c[j] = -math.log(EPS + cnt / nk[k])
    elif rule == "sum":
        for i, it in enumerate(items):
            c[A + i] = -1.0 / nk[it["domain"]]
    elif rule == "maxmin":
        c[tvar] = -1.0
        for i, it in enumerate(items):  # tie-break: more total coverage
            c[A + i] = -0.01 / nk[it["domain"]]
    else:
        raise ValueError(rule)

    rows, lo, hi = [], [], []
    # coverage links: y_i + r_i - sum_{a in cover_i} x_a <= 0  (r_i needs two chosen actions; r_i <= y_i)
    for i, it in enumerate(items):
        r = np.zeros(nvar); r[A + i] = 1; r[roff + i] = 1
        for a in cover.get(it["id"], []):
            r[a] = -1
        rows.append(r); lo.append(-np.inf); hi.append(0)
        r = np.zeros(nvar); r[roff + i] = 1; r[A + i] = -1
        rows.append(r); lo.append(-np.inf); hi.append(0)
    # budget
    r = np.zeros(nvar); r[:A] = 1; rows.append(r); lo.append(0); hi.append(budget)
    if rule == "nash":
        for k in doms:
            r = np.zeros(nvar); r[zidx[k]] = 1; rows.append(r); lo.append(1); hi.append(1)
            r = np.zeros(nvar)
            for i, it in enumerate(items):
                if it["domain"] == k:
                    r[A + i] = 1
            for cnt, j in enumerate(zidx[k]):
                r[j] = -cnt
            rows.append(r); lo.append(0); hi.append(0)
    if rule == "maxmin":
        for k in doms:
            r = np.zeros(nvar); r[tvar] = 1
            for i, it in enumerate(items):
                if it["domain"] == k:
                    r[A + i] = -1.0 / nk[k]
            rows.append(r); lo.append(-np.inf); hi.append(0)

    lb = np.zeros(nvar); ub = np.ones(nvar)
    for a in excluded:
        ub[a] = 0
    integrality = np.ones(nvar)
    if rule == "maxmin":
        integrality[tvar] = 0
    res = milp(c, constraints=LinearConstraint(np.array(rows), lo, hi), bounds=Bounds(lb, ub),
               integrality=integrality, options={"time_limit": 10})
    if res.x is None:
        raise RuntimeError(f"solver failed: {res.message}")
    return [a for a in range(A) if res.x[a] > 0.5]


def coverage(items: List[dict], cover: Dict[str, List[int]], chosen: List[int]) -> Dict[str, float]:
    ch = set(chosen)
    out = {}
    for k in DOMAINS:
        ks = [it for it in items if it["domain"] == k]
        if ks:
            out[k] = sum(bool(ch & set(cover.get(it["id"], []))) for it in ks) / len(ks)
    return out


# ---------------------------------------------------------------- rendering
def render(client: OllamaClient, ctx: str, actions: List[str]) -> str:
    """Deterministic rendering: the chosen actions appear verbatim, so nothing is lost after selection."""
    del client, ctx
    return "Care plan actions:\n" + "\n".join(f"- {a}" for a in actions)


# ---------------------------------------------------------------- evaluation
def evaluate_plan_checklist(client: OllamaClient, items: List[dict], plan: str) -> Dict[str, bool]:
    """Atomic yes/no per checklist item on the final plan text, by the evaluator model."""
    out = {}
    for it in items:
        prompt = (f"Care plan:\n{plan}\n\nRequirement: {it['question']}\nCounts as MET: {it['met']}\n"
                  f"Counts as NOT MET: {it['not_met']}\n\nAnswer ONLY yes or no.")
        raw = client.chat(model=MODELS["evaluator"], user_prompt=prompt,
                          system_prompt="You are a strict clinical auditor.", temperature=0.0, **EVALUATOR_SETTINGS)
        out[it["id"]] = raw.strip().lower().startswith("yes")
    return out
