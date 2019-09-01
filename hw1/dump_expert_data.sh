#!/bin/bash
for entry in "experts"/*
do
  envname=`echo "$entry" | cut -d'/' -f2 | cut -d'.' -f1`
  `python3 run_expert.py $entry $envname --num_rollouts 200 --total_steps 50000`
done
