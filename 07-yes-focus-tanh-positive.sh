#!/usr/bin/env bash

# Add results below and git commit when you have
# e.g. git add experiments/eeeb7a45
#      git commit -m "latest experiment"


pipenv run python -m src.experiments.run \
  --use-attention-focus=True \
  --max-steps=300000 \
  --kb-vector-type='positive' \
  --number-of-questions=5000 \
  --kb-vector-length=12 \
  --kb-list-size=2 \
  --attention-output-activation=tanh

# RESULTS
# accuracy, Model directory
# e.g. 98% eeb7a45/model/25633113
# 98% ffbc02b/model/25633196 might benefit from more training data?
#
#
#
#