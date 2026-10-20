#!/bin/zsh
# Finish the final v2 run: retry failed patients (gpt-oss output limit 2048), then grade with 4 parallel shards per run.
cd "$(dirname "$0")/.."
sed -i '' 's/VERIFIER_SETTINGS = {"num_predict": 1024, "think": "low"}/VERIFIER_SETTINGS = {"num_predict": 2048, "think": "low"}/' scripts/v2_pipeline.py
sed -i '' 's/"gpt-oss:20b", {"num_predict": 1024, "think": "low"}/"gpt-oss:20b", {"num_predict": 2048, "think": "low"}/' scripts/v2_grader.py
for v in none race; do
  for i in 0 1 2; do python3 scripts/v2_run.py --set final --run v2_final --swap $v --shard $i/3 > results/v2_final_retry_${v}_shard$i.log 2>&1 & done
done; wait
for job in "v2_final none" "v1final none" "v2_final race"; do
  set -- ${=job}
  for i in 0 1 2 3; do python3 scripts/v2_grader.py --run $1 --variant $2 --grade --shard $i/4 >> results/v2_final_grade.log 2>&1 & done; wait
done
python3 scripts/v2_grader.py --run v2_final --agreement > results/v2_final_agreement.log 2>&1
echo "final done $(date)" >> results/v2_final_grade.log
