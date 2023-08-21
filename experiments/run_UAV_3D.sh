#!/bin/bash
cd ..;
python3 RunFile.py --application UAV_3D --scenario 3 --two_phase_transient_length 3 --monte_carlo_iterations -1 --horizon 24 --noise_strength_w 0.1 0.1 0.1 0.1 0.1 0.1 --noise_strength_v 0.1 0.1 0.1 --validate_performance 1000;
python3 RunFile.py --application UAV_3D --scenario 3 --two_phase_transient_length 3 --monte_carlo_iterations -1 --horizon 24 --noise_strength_w 1 1 1 1 1 1 --noise_strength_v 0.1 0.1 0.1 --validate_performance 1000;
python3 RunFile.py --application UAV_3D --scenario 3 --two_phase_transient_length 3 --monte_carlo_iterations -1 --horizon 24 --noise_strength_w 0.1 0.1 0.1 0.1 0.1 0.1 --noise_strength_v 1 1 1 --validate_performance 1000;
python3 RunFile.py --application UAV_3D --scenario 3 --two_phase_transient_length 3 --monte_carlo_iterations -1 --horizon 24 --noise_strength_w 1 1 1 1 1 1 --noise_strength_v 1 1 1 --validate_performance 1000;
python3 RunFile.py --application UAV_3D --scenario 3 --two_phase_transient_length 3 --monte_carlo_iterations -1 --horizon 24 --noise_strength_w 1 1 1 1 1 1 --noise_strength_v 0.1 0.1 1 --validate_performance 1000;
python3 RunFile.py --application UAV_3D --scenario 3 --two_phase_transient_length 3 --monte_carlo_iterations -1 --horizon 24 --noise_strength_w 1 1 1 1 1 1 --noise_strength_v 0.1 1 0.1 --validate_performance 1000;