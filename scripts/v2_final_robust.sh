#!/bin/zsh
# Idempotent finisher for the final v2 run. Safe to re-run at any time (after sleep, reboot, or a crash):
# every step appends results and skips work already on disk. Loops until generation and grading are complete.
cd "$(dirname "$0")/.."
sed -i '' 's/VERIFIER_SETTINGS = {"num_predict": 1024, "think": "low"}/VERIFIER_SETTINGS = {"num_predict": 2048, "think": "low"}/' scripts/v2_pipeline.py
sed -i '' 's/"gpt-oss:20b", {"num_predict": 1024, "think": "low"}/"gpt-oss:20b", {"num_predict": 2048, "think": "low"}/' scripts/v2_grader.py
count() { python3 -c "import glob;print(sum(sum(1 for _ in open(f)) for f in glob.glob('results/v2_final_$1_shard*of3.jsonl')))"; }
for attempt in 1 2 3 4 5; do
  [ "$(count none)" -ge 200 ] && [ "$(count race)" -ge 165 ] && break
  for v in none race; do for i in 0 1 2; do
    python3 scripts/v2_run.py --set final --run v2_final --swap $v --shard $i/3 >> results/v2_final_gen_attempts.log 2>&1 &
  done; done; wait
done
echo "generation: none $(count none)/200, race $(count race)/165 at $(date)" >> results/v2_final_grade.log
for job in "v2_final none" "v1final none" "v2_final race"; do
  set -- ${=job}
  until python3 scripts/v2_grader.py --run $1 --variant $2 --check >> results/v2_final_grade.log 2>&1; do   # exits 0 only when nothing is left
    for i in 0 1 2 3; do (python3 scripts/v2_grader.py --run $1 --variant $2 --grade --shard $i/4 >> results/v2_final_grade.log 2>&1) & done
    wait; sleep 60
  done
done
python3 scripts/v2_grader.py --run v2_final --agreement > results/v2_final_agreement.log 2>&1
echo "final done $(date)" >> results/v2_final_grade.log
