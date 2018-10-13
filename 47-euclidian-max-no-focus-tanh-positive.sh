#!/usr/bin/env bash

# Add results below and git commit when you have
# e.g. git add experiments/eeeb7a45
#      git commit -m "latest experiment"


pipenv run python -m src.experiments.run \
  --use-attention-focus=False \
  --score-fn=euclidean \
  --focus-fn=reduce_max \
  --max-steps=300000 \
  --kb-vector-type='positive' \
  --number-of-questions=5000 \
  --kb-vector-length=12 \
  --kb-list-size=2 \
  --attention-output-activation=tanh

# RESULTS
# accuracy, Model directory
# 100%, experiments/3786c2b/model/25657621
#
