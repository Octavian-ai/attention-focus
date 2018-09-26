#!/usr/bin/env bash

# Add results below and git commit when you have
# accuracy, Model directory
# e.g. 98% eeb7a45/model/25633113

pipenv run python -m src.experiments.run \
  --use-attention-focus=True \
  --max-steps=500000 \
  --kb-vector-type='positive' \
  --number-of-questions=5000 \
  --kb-vector-length=12 \
  --kb-list-size=2 \
  --attention-output-activation=abs