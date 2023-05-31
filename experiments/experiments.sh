#!/bin/bash
cd ..;

timeout 3600s python3 RunFile.py --application_id 5 --two_phase_transient_length 4 --monte_carlo_iterations 10 --R_size 12 12 --R_width 1 1 --plot_heatmap 0 1 --horizon 24 --plot_trajectory_2D 0 1;
timeout 3600s python3 RunFile.py --application_id 5 --two_phase_transient_length 4 --monte_carlo_iterations 10 --R_size 24 24 --R_width 0.5 0.5 --plot_heatmap 0 1 --horizon 24 --plot_trajectory_2D 0 1;
timeout 3600s python3 RunFile.py --application_id 5 --two_phase_transient_length 4 --monte_carlo_iterations 10 --R_size 48 48 --R_width 0.25 0.25 --plot_heatmap 0 1 --horizon 24 --plot_trajectory_2D 0 1;

timeout 3600s python3 RunFile.py --application_id 0 --two_phase_transient_length 3 --monte_carlo_iterations -1 --R_size 11 5 11 5 --R_width 2 1.5 2 1.5 --horizon 24 --plot_trajectory_2D 0 2;




timeout 3600s python3 RunFile.py --application_id 5 --two_phase_transient_length 4 --monte_carlo_iterations -1 --R_size 12 12 --R_width 1 1;
timeout 3600s python3 RunFile.py --application_id 5 --two_phase_transient_length 4 --monte_carlo_iterations -1 --R_size 24 24 --R_width 0.5 0.5;
timeout 3600s python3 RunFile.py --application_id 5 --two_phase_transient_length 4 --monte_carlo_iterations -1 --R_size 48 48 --R_width 0.25 0.25;
timeout 3600s python3 RunFile.py --application_id 5 --two_phase_transient_length 4 --monte_carlo_iterations -1 --R_size 96 96 --R_width 0.125 0.125;
