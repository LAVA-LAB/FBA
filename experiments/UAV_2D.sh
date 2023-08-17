#!/bin/bash
cd ..;

python3 RunFile.py --application UAV_2D --two_phase_transient_length 1 --monte_carlo_iterations -1 --R_size 11 5 11 5 --R_width 2 1.5 2 1.5 --horizon 24 --plot_trajectory_2D 0 2 --noise_strength_w 1 1 1 1 --noise_strength_v 1;
python3 RunFile.py --application UAV_2D --two_phase_transient_length 2 --monte_carlo_iterations -1 --R_size 11 5 11 5 --R_width 2 1.5 2 1.5 --horizon 24 --plot_trajectory_2D 0 2 --noise_strength_w 1 1 1 1 --noise_strength_v 1;
python3 RunFile.py --application UAV_2D --two_phase_transient_length 3 --monte_carlo_iterations -1 --R_size 11 5 11 5 --R_width 2 1.5 2 1.5 --horizon 24 --plot_trajectory_2D 0 2 --noise_strength_w 1 1 1 1 --noise_strength_v 1;
python3 RunFile.py --application UAV_2D --two_phase_transient_length 4 --monte_carlo_iterations -1 --R_size 11 5 11 5 --R_width 2 1.5 2 1.5 --horizon 24 --plot_trajectory_2D 0 2 --noise_strength_w 1 1 1 1 --noise_strength_v 1;
python3 RunFile.py --application UAV_2D --two_phase_transient_length 5 --monte_carlo_iterations -1 --R_size 11 5 11 5 --R_width 2 1.5 2 1.5 --horizon 24 --plot_trajectory_2D 0 2 --noise_strength_w 1 1 1 1 --noise_strength_v 1;
python3 RunFile.py --application UAV_2D --two_phase_transient_length 6 --monte_carlo_iterations -1 --R_size 11 5 11 5 --R_width 2 1.5 2 1.5 --horizon 24 --plot_trajectory_2D 0 2 --noise_strength_w 1 1 1 1 --noise_strength_v 1;
python3 RunFile.py --application UAV_2D --two_phase_transient_length 7 --monte_carlo_iterations -1 --R_size 11 5 11 5 --R_width 2 1.5 2 1.5 --horizon 24 --plot_trajectory_2D 0 2 --noise_strength_w 1 1 1 1 --noise_strength_v 1;
python3 RunFile.py --application UAV_2D --two_phase_transient_length 8 --monte_carlo_iterations -1 --R_size 11 5 11 5 --R_width 2 1.5 2 1.5 --horizon 24 --plot_trajectory_2D 0 2 --noise_strength_w 1 1 1 1 --noise_strength_v 1;

python3 RunFile.py --application UAV_2D --two_phase_transient_length 1 --monte_carlo_iterations -1 --R_size 11 5 11 5 --R_width 2 1.5 2 1.5 --horizon 24 --plot_trajectory_2D 0 2 --noise_strength_w 0.1 0.1 0.1 0.1 --noise_strength_v 0.1;
python3 RunFile.py --application UAV_2D --two_phase_transient_length 2 --monte_carlo_iterations -1 --R_size 11 5 11 5 --R_width 2 1.5 2 1.5 --horizon 24 --plot_trajectory_2D 0 2 --noise_strength_w 0.1 0.1 0.1 0.1 --noise_strength_v 0.1;
python3 RunFile.py --application UAV_2D --two_phase_transient_length 3 --monte_carlo_iterations -1 --R_size 11 5 11 5 --R_width 2 1.5 2 1.5 --horizon 24 --plot_trajectory_2D 0 2 --noise_strength_w 0.1 0.1 0.1 0.1 --noise_strength_v 0.1;
python3 RunFile.py --application UAV_2D --two_phase_transient_length 4 --monte_carlo_iterations -1 --R_size 11 5 11 5 --R_width 2 1.5 2 1.5 --horizon 24 --plot_trajectory_2D 0 2 --noise_strength_w 0.1 0.1 0.1 0.1 --noise_strength_v 0.1;
python3 RunFile.py --application UAV_2D --two_phase_transient_length 5 --monte_carlo_iterations -1 --R_size 11 5 11 5 --R_width 2 1.5 2 1.5 --horizon 24 --plot_trajectory_2D 0 2 --noise_strength_w 0.1 0.1 0.1 0.1 --noise_strength_v 0.1;
python3 RunFile.py --application UAV_2D --two_phase_transient_length 6 --monte_carlo_iterations -1 --R_size 11 5 11 5 --R_width 2 1.5 2 1.5 --horizon 24 --plot_trajectory_2D 0 2 --noise_strength_w 0.1 0.1 0.1 0.1 --noise_strength_v 0.1;
python3 RunFile.py --application UAV_2D --two_phase_transient_length 7 --monte_carlo_iterations -1 --R_size 11 5 11 5 --R_width 2 1.5 2 1.5 --horizon 24 --plot_trajectory_2D 0 2 --noise_strength_w 0.1 0.1 0.1 0.1 --noise_strength_v 0.1;
python3 RunFile.py --application UAV_2D --two_phase_transient_length 8 --monte_carlo_iterations -1 --R_size 11 5 11 5 --R_width 2 1.5 2 1.5 --horizon 24 --plot_trajectory_2D 0 2 --noise_strength_w 0.1 0.1 0.1 0.1 --noise_strength_v 0.1;
