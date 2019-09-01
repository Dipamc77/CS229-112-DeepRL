#!/bin/bash
for entry in "experts"/*
do
  envname=`echo "$entry" | cut -d'/' -f2 | cut -d'.' -f1`
  `python3 run_imitation.py $envname &`
  `python3 run_imitation.py $envname --l2_reg 1e-3 &`
  `python3 run_imitation.py $envname --batchnorm &`
done
