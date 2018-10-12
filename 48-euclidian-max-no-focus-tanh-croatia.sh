#!/usr/bin/env bash

# Add results below and git commit when you have
# e.g. git add experiments/eeeb7a45
#      git commit -m "latest experiment"


pipenv run python -m src.experiments.run \
  --use-attention-focus=False \
  --score-fn=euclidean \
  --focus-fn=reduce_max \
  --max-steps=300000 \
  --kb-vector-type='croatia' \
  --number-of-questions=2000 \
  --kb-vector-length=300 \
  --kb-list-size=77 \
  --attention-output-activation=tanh \
  --finder-initial-lr=1.1e-5 \
  --enable-lr-finder=False

# RESULTS
# accuracy, Model
# 100%