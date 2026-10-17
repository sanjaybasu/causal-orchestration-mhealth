#!/bin/zsh
# Development round 2: wait for v1 generation to finish, run all 150 dev patients in 3 shards, then grade with the panel.
cd "$(dirname "$0")/.."
while pgrep -f "run_r1_three_arm.py --run-name r1_three_arm_n200" >/dev/null; do sleep 300; done
for i in 0 1 2; do python3 scripts/v2_run.py --set dev --run v2_dev2 --shard $i/3 > results/v2_dev2_shard$i.log 2>&1 & done
wait
python3 scripts/v2_grader.py --run v2_dev2 --grade > results/v2_dev2_grade.log 2>&1
python3 scripts/v2_grader.py --run v2_dev2 --agreement > results/v2_dev2_agreement.log 2>&1
echo "dev2 done $(date)" >> results/v2_dev2_grade.log
