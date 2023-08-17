#!/bin/bash
cd ..;

python3 RunFile.py --application spacecraft --two_phase_transient_length 4 --monte_carlo_iterations -1 --R_size 11 11 5 5 --R_width 2 2 0.05 0.05 --horizon 32 --plot_heatmap 0 1 --plot_trajectory_2D 0 1 --noise_strength_w 1 1 1 1 --noise_strength_v 1 1;