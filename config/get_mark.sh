set -ex

export TARGET_SCORE=$(head -2 $1/target_score.txt | tail -1)

echo ${TARGET_SCORE}
