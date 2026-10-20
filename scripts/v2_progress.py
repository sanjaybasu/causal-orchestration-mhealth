#!/usr/bin/env python3
"""One-line progress for the final v2 run."""
import glob, time
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "results"
lines = lambda pat: sum(sum(1 for _ in open(f)) for f in glob.glob(str(R / pat)))
fails = sum(open(f).read().count("FAIL") for f in glob.glob(str(R / "v2_final_*_shard*.log")))
done = (R / "v2_final_grade.log").exists() and "final done" in (R / "v2_final_grade.log").read_text()
print(f"{time.strftime('%a %H:%M')} | final {lines('v2_final_none_shard*of3.jsonl')}/200 | race-swap {lines("v2_final_race_shard*of3.jsonl")}/165 | "
      f"failures {fails} | graded: v2 {lines('v2_final_panel.jsonl')}, v1 {lines('v1final_panel.jsonl')}, race {lines('v2_final_race_panel.jsonl')} | done {done}")
