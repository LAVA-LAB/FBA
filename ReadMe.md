# Introduction of this ReadMe file

This artefact contains the source code for paper submission 10143 to NeurIPS 2021, with the title:

**<center>Sampling-Based Robust Control of Autonomous Systems with Non-Gaussian Noise</center>**

The files in this repository contain everything that is needed to replicate the results presented in the paper. Our simulations ran on a Linux machine with 32 3.7GHz cores and 64 GB of RAM. Using the instructions below, the experiments may be replicated on a virtual machine, or on your own machine.

Python version: `3.8.3`. For a list of the required Python packages, please see the `requirements.txt` file. 

------

# Table of contents

[TOC]

------

# Installation and execution

We tested the artefact with the *TACAS 21 Artefact Evaluation VM - Ubuntu 20.04 LTS*, which can be downloaded from the links specified in steps 1 and 2 below. Please follow the instructions below if you want to use the artefact within a virtual machine. 

If you plan to use the program directly on your own machine, you may skip steps 1 and 2.

**<u>Important note:</u>** the PRISM version that we use only runs on MacOS or Linux.

## 1. Download and install VirtualBox

To use the artefact on a virtual machine, first download and install the *VirtualBox* host from the following webpage:

https://www.virtualbox.org/wiki/Downloads

The artefact has been tested with *VirtualBox 6.1.18* on *Windows 10*. 

## 2. Download the TACAS 21 Virtual Machine

We tested the artefact on the *TACAS 21 Artefact Evaluation VM*, which can be downloaded from Zenodo.org:

https://zenodo.org/record/4041464

Download the `TACAS 21.ova` file (size 3.6 GB), then open the VirtualBox application, and import this file by clicking `File` -> `Import Virtual Appliance`. In this menu, select the `TACAS 21.ova` file and click `next`. In the menu that follows, you may change the assigned RAM memory. 

**<u>Note:</u>** in order to run the larger applications, you may want to increase the RAM memory, e.g. to 8192 MB.

After setting the desired settings, click `import` to import the appliance (this may take a few minutes). When this is finished, boot up the virtual machine.

Note that other virtual machines that support Python 3 may work as well, but we tested the artefact specifically for this one.

## 3. Install dependencies

In addition to Python 3 (which is installed on the TACAS 21 virtual machine by default), a number of dependencies must be installed on your (virtual) machine:

1. Git - Can be installed using the command:

   ```bash
   $ sudo apt install git
   ```

   **<u>Note:</u>** when asked for a password, the default login and password are `tacas21` / `tacas21`.

2. Java Development Kit (required to run PRISM) - Can be installed using the commands:

   ```bash
   $ sudo apt update 
   $ sudo apt install default-jdk
   ```

3. PRISM (iMDP branch) - In the desired PRISM installation folder, run the following commands:

   ```bash
   $ git clone -b imc https://github.com/davexparker/prism prism-imc
   $ cd prism-imc/prism; make
   ```

   For more details on using PRISM, we refer to the PRISM documentation on 
   https://www.prismmodelchecker.org

## 4. Copy artefact files and install packages

Download and extract the artefact files to a folder on the virtual machine (or on your own machine, if you are not using a virtual one) with writing access (needed to store results).

Open a terminal and navigate to the artefact folder. Then, run the following command to install the required packages:

```bash
$ pip3 install -r requirements.txt
```

The following packages will be installed:

- matplotlib==3.2.2
- numpy==1.18.5
- pandas==1.0.5
- pyopengl==3.1.5
- scipy==1.5.0
- seaborn==0.10.1
- xlrd==1.2.0
- xlsxwriter==1.2.9
- visvis==1.12.4

## 5. Set default folders and options

To ensure that PRISM can be found by the script, **you need to modify the PRISM folder** in the  `options.txt` file. Set the PRISM folder to the one where you installed it (the filename should end with `/prism/`, such that it points the folder in which the `bin/` folder is located), and save your changes.

If desired, you may also make other changes in the configuration of the script in the `options.txt` file. An overview of the most important settings is given below:

- `scenarios.samples` : the number of samples the script uses in the first iteration
- `scenarios.samples_max` : the number of samples after which the iterative scheme is terminated
- `scenarios.confidence` : the confidence level used for computing transition probability intervals
- `mdp.prism_folder` : folder where PRISM is located; should end with `/prism/` (the folder in which the `bin/` folder is located)
- `mdp.mode` : if “*interval*”, an interval MDP is created. If “*estimate*”, a regular MDP is created
- `mdp .prism_model_writer` : if “*explicit*”, a PRISM model is created in explicit form. If “*default*”, a standard PRISM model is created. See the PRISM documentation for more details.
- `mdp.prism_java_memory` : the memory allocated to JAVA when running PRISM. The default value is 1 GB, but when solving large models, this may be increased (e.g. to 8 GB).
- `main.iterative` : if True, the iterative scheme is enabled; if False, it is disabled

## 6. Run the script

Run the `SBA-RunFile.py` file to execute the program, by typing the command:

```bash
$ python RunFile.py
```

You will be asked to make a number of choices:

1. The **application** (i.e. model) you want to work with (see below for details on how to add a model). For some applications, you will be asked an additional question, such as the dimension (for the UAV case) and grid size (for the 1-zone BAS case).
2. Whether you want to run **Monte Carlo** simulations. If chosen to do so, you are asked an additional question to fill in the **number of Monte Carlo simulations** to run.
3. Whether you want to **start a new abstraction** or **load existing PRISM results**.

The user can recreate the results presented in the paper, by choosing the **3D UAV** application or the **2-zone building automation system (BAS)** application.

**<u>Important note:</u>** after every iteration of the 3D UAV case, an interactive 3D plot is created with `visvis`, that shows a number of trajectories under the optimal controller. **<u>This plot will pause the script</u>**, until it is closed manually by the user. Note that a screenshot of the plot is automatically stored in the results folder of the current iteration.

## 7. Inspect the results 

All results are stored in the `output/` folder. When running `SBA-RunFile.py` for a new abstraction, a new folder is created that contains the application name and the current datetime, such as `ScAb_UAV_06-02-2021_13-46-29/`.

Results that apply to all iterations of the iterative abstraction scheme are saved directly in this folder. This includes an Excel file describing all model sizes, run times, etc.

For every iteration, a subfolder is created based on the number of samples that was used, e.g. `N=3200`. Within this subfolder, all results specific to that single iteration are saved. This includes:

- The PRISM model files. Depending on the mode, this can either be in explicit format (in which case a `.lab`, `.sta`, and `.tra` file are created), or as a single file if the default mode is selected.
- An Excel file that describes all results, such as the optimal policy, model size, run times, etc., of the current iteration.
- Various plots, showing the appropriate results for the current iteration.

------

# Adding or modifying model definitions

You can add models or change the existing ones by modifying the file `core/modelDefinitions.py`. Every application is defined as a class, having the application name as the class name. An application class has two functions:

### Initializing function

Here, the basic setup of the application is defined. This includes the following elements:

- `setup['deltas']` (list of integers) is assigned a list of integers (with only one value by default). It denotes the number of discrete time steps that are grouped together, to render the dynamical system fully actuated. For example, if the dimension of the state is 6, and the dimension of the control input is 3, then it will be `setup['deltas'] = [2]`.
- `setup['control']['limits'] ['uMin'] / ['uMax']` (list of integers) are the control authority limits. It is given as a list, and every entry reflects a dimension of the state.
- `setup['partition']['nrPerDim']` (list of integers) is the number of regions defined in every dimension of the state space. Note that the partition has a region centred at the origin when odd values are given.
- `setup['partition']['width']` (list of floats) is the width of the regions in every dimension of the state space.
- `setup['partition']['origin']` (list of floats) denotes the origin of the partitioned region of the state space. 
- `setup['targets']['nrPerDim']` (either `'auto'` or a list of integers) is the number of target points defined in every dimension of the state. When `'auto'` is given, the number of target points equals the number of regions of the partition.
- `setup['targets']['domain']` (either `'auto'` or a list of integers) is the domain over which target points are defined. When `'auto'` is given, the domain is set equal to that of the state space partition.
- `setup['specification']['goal']` (nested list of floats) is the list of points whose associated partitioned regions are in the goal region. The function `setStateBlock()` is a helper function to easily define blocks of goal regions, by creating slices in the state space partition in given dimensions.
- `setup['specification']['critical']` (nested list of floats) is the list of points whose associated partitioned regions are in the critical region. The function `setStateBlock()` is a helper function to easily define blocks of critical regions, by creating slices in the state space partition in given dimensions.
- `tau` (float) is the time discretization step size.
- `setup['endTime']` (integer > 0) is the finite time horizon over which the reach-avoid problem is solved.

### SetModel function

In the `setModel` function, the linear dynamical system is defined in the following form (see the submitted paper for details): 
$$
\mathbf{x}_{k+1} = A \mathbf{x}_k + B \mathbf{u}_k + \mathbf{q}_k + \mathbf{w}_k,
$$
where:

- `A` is an n x n matrix.
- `B` is an n x p matrix.
- `Q` is a n x 1 column vector that reflects the additive deterministic disturbance (q-term in the equation above).
- If Gaussian noise is used, `noise['w_cov']` is the covariance matrix of the w-term in the equation above. Note that non-Gaussian noise (from the Dryden gust model) is used for the UAV case.

Note that is the current version of the codes is not compatible (yet) with partial observability (i.e. defining an observer). Thus, make sure to set the argument `observer = False`.

For some models, the model definition is given in non-discretized form, i.e.
$$
\dot{\mathbf{x}}(t) = A_c\mathbf{x}(t) + B_c\mathbf{u}(t) + \mathbf{q}_c(t) + \mathbf{w}_c(t),
$$
where subscript c indicates that these matrices and vectors differ from the ones above. If a continuous-time dynamical model is given, it is discretized using one of two methods:

- Using a forward Euler method.
- Using a Gears discretization method.

