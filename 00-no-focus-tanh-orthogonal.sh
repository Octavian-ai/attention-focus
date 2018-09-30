#!/usr/bin/env bash

# Add results below and git commit when you have
# e.g. git add experiments/eeeb7a45
#      git commit -m "latest experiment"


pipenv run python -m src.experiments.run \
  --use-attention-focus=False \
  --max-steps=300000 \
  --kb-vector-type='orthogonal' \
  --number-of-questions=5000 \
  --kb-vector-length=12 \
  --kb-list-size=2 \
  --attention-output-activation=tanh

# RESULTS
# accuracy, Model directory
# e.g. 98% eeb7a45/model/25633113
# 37% df42991/model/25633138
# 58.75% 09fc24e/model/25637712
#
#
#
