#!/usr/bin/env python3
"""
Write every prompt used in the three-arm experiment to markdown, by calling the
production functions with a recording client. The S1 Appendix section on prompts
is generated from this output, so the appendix cannot drift from the code.

  python scripts/r1_dump_prompts.py > results/r1_analysis/prompts_verbatim.md
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_r1_three_arm as r1  # noqa: E402
from run_ollama_open_source_experiments import evaluate_plan, generate_initial_plan, generate_role_critique  # noqa: E402

PATIENT = "{PATIENT_CONTEXT}"
DRAFT = "{CURRENT_DRAFT}"


class Recorder:
    def __init__(self):
        self.calls = []

    def chat(self, model, user_prompt, system_prompt=None, num_predict=512, temperature=0.2, seed=None):
        self.calls.append(dict(model=model, system=system_prompt, user=user_prompt,
                               num_predict=num_predict, temperature=temperature, seed=seed))
        return '{"safety":0.5,"efficiency":0.5,"equity":0.5,"composite":0.5}'


def block(title, c):
    seed = f", seed {c['seed']}" if c["seed"] is not None else ""
    return (f"#### {title}\n\nModel: `{c['model']}`; temperature {c['temperature']}{seed}; "
            f"maximum output tokens {c['num_predict']}.\n\nSystem prompt:\n\n```\n{c['system']}\n```\n\n"
            f"User prompt:\n\n```\n{c['user']}\n```\n")


def main():
    rec = Recorder()
    out = ["## Prompts (generated from code)\n",
           "`{PATIENT_CONTEXT}` is the patient context block, `{CURRENT_DRAFT}` the current plan, "
           "and `{CRITIQUE_n}` a critique text.\n"]

    generate_initial_plan(rec, r1.MODELS["generator"], PATIENT)
    out.append(block("Draft (all arms)", rec.calls[-1]))
    for role in r1.ROLE_ORDER:
        generate_role_critique(rec, r1.MODELS[role], role, PATIENT, DRAFT)
        out.append(block(f"{role.capitalize()} critique (role-specialized Nash and additive arms)", rec.calls[-1]))
    r1.critiques_for_arm(rec, "B", PATIENT, DRAFT)
    out.append(block("Generic critique (generic self-critique arm; called three times per round)", rec.calls[-1]))
    r1.synthesize(rec, PATIENT, DRAFT, ["{CRITIQUE_1}", "{CRITIQUE_2}", "{CRITIQUE_3}"])
    out.append(block("Synthesis, rounds 1 and 2 (all arms)", rec.calls[-1]))
    r1.synthesize(rec, PATIENT, DRAFT, ["{CRITIQUE_1}", "{CRITIQUE_2}", "{CRITIQUE_3}"],
                  temperature=r1.CANDIDATE_TEMPERATURE, seed=1)
    out.append(block("Candidate synthesis, round 3 (all arms; seeds 1, 2, 3)", rec.calls[-1]))
    evaluate_plan(rec, r1.MODELS["outcome_judge"], PATIENT, "{CARE_PLAN}")
    out.append(block("Judge (selection judge llama3.1:8b and outcome judges use the same prompt)", rec.calls[-1]))
    print("\n".join(out))


if __name__ == "__main__":
    main()
