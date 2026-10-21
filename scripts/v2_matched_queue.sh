#!/bin/zsh
# After the final grading finishes: compute-matched refine arm (3 shards, resumable), then panel grading of it.
cd "$(dirname "$0")/.."
until grep -q "final done" results/v2_final_grade.log 2>/dev/null; do sleep 600; done
count() { python3 -c "import glob;print(sum(sum(1 for _ in open(f)) for f in glob.glob('results/v2_final_matchedR_shard*of3.jsonl')))"; }
until [ "$(count)" -ge 200 ]; do
  for i in 0 1 2; do python3 scripts/v2_refine_matched.py $i/3 >> results/v2_matchedR_gen.log 2>&1 & done; wait; sleep 60
done
until python3 scripts/v2_grader.py --run v2_final --variant matchedR --check >> results/v2_matchedR_grade.log 2>&1; do
  for i in 0 1 2 3; do (python3 scripts/v2_grader.py --run v2_final --variant matchedR --grade --shard $i/4 >> results/v2_matchedR_grade.log 2>&1) & done; wait; sleep 60
done
echo "matched done $(date)" >> results/v2_matchedR_grade.log
