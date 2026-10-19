#!/bin/zsh
# Final v2 run on the 200 patients: unswapped, then race-swapped; panel grading of both and of the v1 plans.
cd "$(dirname "$0")/.."
for i in 0 1 2; do python3 scripts/v2_run.py --set final --run v2_final --shard $i/3 > results/v2_final_none_shard$i.log 2>&1 & done; wait
for i in 0 1 2; do python3 scripts/v2_run.py --set final --run v2_final --swap race --shard $i/3 > results/v2_final_race_shard$i.log 2>&1 & done
python3 scripts/v2_grader.py --run v2_final --grade > results/v2_final_grade.log 2>&1
python3 scripts/v2_grader.py --run v1final --grade > results/v1final_grade.log 2>&1
wait
python3 scripts/v2_grader.py --run v2_final --variant race --grade >> results/v2_final_grade.log 2>&1
python3 scripts/v2_grader.py --run v2_final --agreement > results/v2_final_agreement.log 2>&1
echo "final done $(date)" >> results/v2_final_grade.log
