#!/bin/zsh
# Development round 2: wait for v1 generation to finish, run all 150 dev patients in 3 shards, then grade with the panel.
cd "$(dirname "$0")/.."
true
for i in 0 1 2; do python3 scripts/v2_run.py --set dev --run v2_dev3 --shard $i/3 > results/v2_dev3_shard$i.log 2>&1 & done
wait
python3 scripts/v2_grader.py --run v2_dev3 --grade > results/v2_dev3_grade.log 2>&1
python3 scripts/v2_grader.py --run v2_dev3 --agreement > results/v2_dev3_agreement.log 2>&1
echo "dev3 done $(date)" >> results/v2_dev3_grade.log
