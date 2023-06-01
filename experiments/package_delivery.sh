#!/bin/bash
cd ..;

python3 RunFile.py --application_id 5 --two_phase_transient_length 4 --monte_carlo_iterations 100 --R_size 12 12 --R_width 1 1 --plot_heatmap 0 1 --horizon 24 --plot_trajectory_2D 0 1;
python3 RunFile.py --application_id 5 --two_phase_transient_length 4 --monte_carlo_iterations 100 --R_size 24 24 --R_width 0.5 0.5 --plot_heatmap 0 1 --horizon 24 --plot_trajectory_2D 0 1;
python3 RunFile.py --application_id 5 --two_phase_transient_length 4 --monte_carlo_iterations 100 --R_size 48 48 --R_width 0.25 0.25 --plot_heatmap 0 1 --horizon 24 --plot_trajectory_2D 0 1;
