import pandas as pd
import numpy as np

import argparse
import pandas as pd
import os
import sys
import numpy as np
from datetime import datetime
from pathlib import Path

def simplify_instance(instance):
    print('Try to simplify instance:', instance)

    dic = {
        'brp16_2': 'BRP (16,2)',
        'brp32_3': 'BRP (32,3)',
        'brp64_4': 'BRP (64,4)',
        'brp512_5': 'BRP (512,5)',
        'brp1024_6': 'BRP (1024,6)',
        'crowds3_5': 'Crowds (3,5)',
        'crowds6_5': 'Crowds (6,5)',
        'crowds10_5': 'Crowds (10,5)',
        'nand2_4': 'NAND (2,4)',
        'nand5_10': 'NAND (5,10)',
        'nand10_15': 'NAND (10,15)',
        'virus': 'Virus',
        'wlan0_param': 'WLAN0',
        'csma2_4_param': 'CSMA (2,4)',
        'coin4': 'Coin (4)',
        'maze_simple_extended_m5': 'Maze',
        'pomdp_drone_4-2-mem1-simple': 'Drone (mem1)',
        'pomdp_drone_4-2-mem5-simple': 'Drone (mem5)',
        'pomdp_satellite_36_sat_5_act_0_65_dist_5_mem_06': 'Satellite (36,5)',
        'pomdp_prob_36_sat_065_dist_1_obs_diff_orb_len_40': 'Satellite(36,65',
    }

    for d, v in dic.items():
        if d in instance:
            simplified = v
            return simplified

    print('>>> Warning: Could not simplify instance name "{}"'.format(instance))
    return instance


def find_xlsx_files(folder_path):
    xlsx_files = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".xlsx"):
                xlsx_files.append(os.path.join(root, file))

    xlsx_files.sort(key=lambda x: os.path.getmtime(x))

    return xlsx_files

def load_files(mypath):

    # Get all files in provided folder
    filenames = find_xlsx_files(mypath)

    print('- Nr of files found: {}'.format(len(filenames)))
    if len(filenames) == 0:
        print('>> No files found.')
        sys.exit()

    df = {}

    for i, file in enumerate(filenames):

        print('-- Read file "{}"'.format(file))
        data  = {'model_size': pd.read_excel(os.path.join(mypath, file), sheet_name="Model size"),
                 'performance': pd.read_excel(os.path.join(mypath, file), sheet_name="Performance"),
                 'run_time': pd.read_excel(os.path.join(mypath, file), sheet_name="Run time")
                 }

        abstraction_time = data['run_time']['0_init'][0] + data['run_time']['1_partition'][0] + \
            data['run_time']['2_enabledActions'][0] + data['run_time']['3_probabilities'][0]

        verification_time = data['run_time']['5_MDPsolved'][0]

        df[i] = pd.Series({
            'Model':                        Path(file).parent.name,
            'Instance':                     '',
            'n (state dimension)':          '',
            'bar{N}':                       '',
            'States':                       data['model_size']['States'][0],
            'Transitions':                  data['model_size']['Transitions'][0],
            'Abstraction':                  abstraction_time,
            'Compute pi*':                  verification_time,
            'Sat. prob. bound (eta*)':     data['performance']['PRISM reachability'][0],
            'Monte Carlo (bar{p})':         data['performance']['Empirical reachability'][0]
        })

    df_merged = pd.concat(df, axis=1).T

    return df_merged


if __name__ == "__main__":

    pd.set_option('display.max_columns', 100)

    dt = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    parser = argparse.ArgumentParser(description="Export results to tables as in paper")

    # Path to PRISM model to load
    parser.add_argument('--folder', type=str, action="store", dest='folder',
                        default='output/', help="Folder to combine output files from")
    parser.add_argument('--table_name', type=str, action="store", dest='table_name',
                        default='output/export_{}'.format(dt), help="Name of table csv file")

    args = parser.parse_args()

    root_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
    mypath = os.path.join(root_dir, args.folder)

    mypath = '/Users/tbadings/Documents/ssh_sync/2023-08-25/'

    print('- Path to search:', mypath)
    df_out = load_files(mypath)

    #####

    df_out.to_csv(os.path.join(root_dir, args.table_name + '.csv'))

    # Round entries for tex table
    df_out['Abstraction'] = df_out['Abstraction'].map('{:,.1f}'.format)
    df_out['Compute pi*'] = df_out['Compute pi*'].map('{:,.1f}'.format)
    df_out['Sat. prob. bound (eta*)'] = df_out['Sat. prob. bound (eta*)'].map('{:,.3f}'.format)
    df_out['Monte Carlo (bar{p})'] = df_out['Monte Carlo (bar{p})'].map('{:,.3f}'.format)

    df_out.style.to_latex(os.path.join(root_dir, args.table_name + '.tex'))

    print('- Exported to CSV and LaTeX table')