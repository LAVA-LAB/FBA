# Introduction of this ReadMe file

This artefact contains the source code for the paper:

 [Thom Badings, Hasan Poonawala, Marielle Stoelinga & Nils Jansen (2023). Correct-by-construction reach-avoid control of partially observable linear stochastic systems. ArXiv preprint.](https://arxiv.org/abs/2103.02398)

This repository contains all code and instructions that are needed to replicate the results presented in the paper. The experiments run single-threaded on a computer with a 4GHz Intel Core i9 CPU and 32 GB of RAM.

Python version: `3.8.3`. For a list of the required Python packages, please see the `requirements.txt` file. 

------



# Installation and reproducing experiments

We have tested the artefact on MacOS Ventura 13.4.1, and on Ubuntu 22.04.2.

## 3. Install dependencies

In addition to Python, a number of dependencies must be installed on your machine:

1. Git - Can be installed using the command:

   ```bash
   $ sudo apt update 
   $ sudo apt install git
   ```
   
2. Java Development Kit (required to run PRISM) - Can be installed using the commands:

   ```bash
   $ sudo apt install default-jdk
   ```

3. PRISM (iMDP branch) - In the desired PRISM installation folder, run the following commands:

   ```bash
   $ git clone -b imc https://github.com/davexparker/prism prism-imc
   $ cd prism-imc/prism; make
   ```

   For more details on using PRISM, we refer to the PRISM documentation on 
   https://www.prismmodelchecker.org

4. To compute the error bound by which goal and critical regions are augmented, we rely on semi-definite programming. To this end, the blas+lapack library is required. This library is installed using the following command:

   ```bash
   $ sudo apt install -y libatlas-base-dev
   ```

5. To create the 3D UAV trajectory plots, you may need to install a number of libraries requires for Qt, which can be done using the command:

   ```bash
   $ sudo apt-get install -y libdbus-1-3 libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0                          libxcb-render-util0 libxcb-xinerama0 libxcb-xinput0 libxcb-xfixes0
   ```

## 4. Copy artefact files and install packages

Download and extract the artefact files to a desired folder on your machine. Then, run the following command in the folder where you extracted the artefact to install the required packages:

```bash
$ pip3 install -r requirements.txt
```

For a full list of all packages installed, please see `requirements.txt`.

**<u>Note:</u>** It can be critical to install the headless version of opencv. This is important because both cv2 (opencv) and pyqt5 come with a version of the XCB plugin, but these versions are incompatible.

## 5. Set default folders and options

To ensure that PRISM can be found by the script, **you need to modify the PRISM folder** in the  `options.txt` file. Set the PRISM folder to the one where you installed it (the filename should end with `/prism/`, such that it points to the folder in which the `bin/` folder is located), and save your changes. For example, this line of the `options.txt` file can look like this:

```
mdp.prism_olfder = /home/<location-to-prism>/prism-imc/prism/
```

If desired, you may also make other changes in the configuration of the script in the `options.txt` file. An overview of the most important settings is given below:

- `mdp.prism_folder` : folder where PRISM is located; should end with `/prism/` (the folder in which the `bin/` folder is located)
- `mdp.mode` : if “*interval*” (default value), an interval MDP is created. If “*estimate*”, a regular MDP is created
- `mdp.prism_java_memory` : the memory allocated to Java when running PRISM. The default value is 2 GB, but when solving large models, this may be increased (the benchmarks in the paper all ran on a machine with 32 GB of memory allocated to Java).
- `plotting.3D_UAV` : if True, the 3D plots for the 3D UAV benchmark are created. Note that **<u>this plot pauses the script until it is closed</u>**. If you do not want this behavior, you need to disable this option.

## 6. Run the script

The `RunFile.py` file, which executes the main program, expects a number of arguments. For examples, see the shell scripts in the `experiments/` folder. For example, the following commands runs the 2D UAV benchmark, with the two-phase time horizon with a transient phase of length 3:

```
$ python3 RunFile.py --application UAV_2D --two_phase_transient_length 3 --monte_carlo_iterations -1 --R_size 11 5 11 5 --R_width 2 1.5 2 1.5 --horizon 24 --plot_trajectory_2D 0 2 --noise_strength_w 1 1 1 1 --noise_strength_v 1 --validate_performance 1000;
```

To reproduce all the results presented in the experiments in the paper, navigate to the `experiments/` folder and run the following command:

```bash
$ bash run_all_experiments.sh
```

## 7. Inspect the results 

All results are stored in the `output/` folder. When running a new benchmark instance, a new folder is created that contains the application name and the current datetime, such as `FiAB_<model>_<more options>_<date>/`.

Within this folder, all results specific to that single iteration are saved. This includes:

- The PRISM model files in explicit format (a `.lab`, `.sta`, and `.tra` and `.pctl` file are created).
- An Excel file that describes all results, such as the optimal policy, model size, run times, etc., of the current iteration.
- Various plots, showing the appropriate results for the current iteration.
