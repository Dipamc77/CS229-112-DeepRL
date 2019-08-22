#!/bin/bash
for entry in "experts"/*
do
  envname=`echo "$entry" | cut -d'/' -f2 | cut -d'.' -f1`
  for seed in {1..5}
  do
    `python3 run_expert.py $entry $envname --num_rollouts 200 --seed $seed --total_steps 20000`
  done
done
