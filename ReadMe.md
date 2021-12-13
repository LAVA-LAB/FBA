# Introduction of this ReadMe file

This artefact contains the source code for the paper:

**<center>Filter-Based Abstractions for Safe Planning of Partially Observable Dynamical Systems</center>**

The files in this repository contain everything that is needed to replicate the results presented in the paper. Our simulations ran on a Linux machine with 32 3.7GHz cores and 64 GB of RAM. Using the instructions below, the experiments may be replicated on a virtual machine, or on your own machine.

Python version: `3.8.3`. For a list of the required Python packages, please see the `requirements.txt` file. 

------



# Table of contents

[TOC]

------



# Installation and execution of the main program

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

**<u>Note:</u>** In order to run the larger applications, you may want to increase the RAM memory, e.g. to 8192 MB.

After setting the desired settings, click `import` to import the appliance (this may take a few minutes). When this is finished, boot up the virtual machine.

Note that other virtual machines that support Python 3 may work as well, but we tested the artefact specifically for this one.

## 3. Install dependencies

In addition to Python 3 (which is installed on the TACAS 21 virtual machine by default), a number of dependencies must be installed on your (virtual) machine:

1. Git - Can be installed using the command:

   ```bash
   $ sudo apt update 
   $ sudo apt install git
   ```

   **<u>Note:</u>** When asked for a password, the default login and password are `tacas21` / `tacas21`.

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

Download and extract the artefact files to a folder on the virtual machine (or on your own machine, if you are not using a virtual one) with writing access (needed to store results).

Open a terminal and navigate to the artefact folder. Then, run the following command to install the required packages:

```bash
$ pip3 install -r requirements.txt
```

The following packages will be installed:

- cvxopt==1.2.7
- cvxpy==1.1.15
- opencv-python-headless==4.5.3.56
- imageio==2.9.0 (needed for visvis)
- matplotlib==3.3.4
- numpy==1.20.1
- pandas==1.3.2
- polytope==0.2.3
- pyopengl==3.1.5 (needed for visvis)
- pyqt5==5.15.4 (needed for visvis)
- scipy==1.6.2
- seaborn==0.11.2
- visvis==1.12.4 (to create 3D UAV plots)
- xlrd==2.0.1
- xlsxwriter==3.0.1

**<u>Note:</u>** It can be critical to install the headless version of opencv. This is important, because both cv2 (opencv) and pyqt5 come with a version of the XCB plugin, but these versions are incompatible.

## 5. Set default folders and options

To ensure that PRISM can be found by the script, **you need to modify the PRISM folder** in the  `options.txt` file. Set the PRISM folder to the one where you installed it (the filename should end with `/prism/`, such that it points the folder in which the `bin/` folder is located), and save your changes. For example, this line of the `options.txt` file can look like this:

```
/home/<location-to-prism>/prism-imc/prism/
```

If desired, you may also make other changes in the configuration of the script in the `options.txt` file. An overview of the most important settings is given below:

- `mdp.prism_folder` : folder where PRISM is located; should end with `/prism/` (the folder in which the `bin/` folder is located)
- `mdp.mode` : if “*interval*” (default value), an interval MDP is created. If “*estimate*”, a regular MDP is created
- `mdp.prism_java_memory` : the memory allocated to Java when running PRISM. The default value is 2 GB, but when solving large models, this may be increased (the benchmarks in the paper all ran on a machine with 32 GB of memory allocated to Java).
- `plotting.partitionPlot` : if True, a 2D plot of the partition is created; if False, this plot is not created
- `plotting.3D_UAV` : if True, the 3D plots for the 3D UAV benchmark are created. Note that **<u>this plot pauses the script until it is closed</u>**. If you do not want this behaviour, you need to disable this option.

## 6. Run the script

Run the `RunFile.py` file to execute the program, by typing the command:

```bash
$ python RunFile.py
```

You will be asked to make a number of choices:

1. The **application** (i.e. benchmark) you want to work with (see below for details on how to add a model). For some applications, you will be asked an additional question, such as the dimension (for the UAV case) and grid size (for the 1-zone BAS case).
   - **<u>Note 1:</u>** The 3D UAV and 2-zone BAS applications are quite memory intense, and could take about an hour to run (depending on your machine). If you want to run a smaller case, please consider choosing the 2D UAV or 1-zone BAS application.
   - **<u>Note 2:</u>** If you experience problems with creating the **3D trajectory plots** for the 3D UAV application, you can disable it by setting `plotting.3D_UAV = False` in the `options.txt` file (see Section 5).
2. Whether you want to run **Monte Carlo** simulations. If chosen to do so, you are asked an additional question to fill in the **number of Monte Carlo simulations** to run.
3. Whether you want to **start a new abstraction** or **load existing PRISM results**.

The user can recreate the results presented in the paper, by choosing the **Double Integrator**, **2D UAV**, or the **3D UAV** application.

**<u>Important note:</u>** After every iteration of the 3D UAV case, an interactive 3D plot is created with `visvis`, that shows a number of trajectories under the optimal controller. **<u>This plot may pause the script</u>**, until it is closed manually by the user.

## 7. Inspect the results 

All results are stored in the `output/` folder. When running `RunFile.py` for a new abstraction, a new folder is created that contains the application name and the current datetime, such as `FiAB_<application>_<more options>_<date>/`.

Within this folder, all results specific to that single iteration are saved. This includes:

- The PRISM model files in explicit format (a `.lab`, `.sta`, and `.tra` and `.pctl` file are created).
- An Excel file that describes all results, such as the optimal policy, model size, run times, etc., of the current iteration.
- Various plots, showing the appropriate results for the current iteration.

------



# Adding or modifying model definitions

You can add models or change the existing ones by modifying the file `core/modelDefinitions.py`. Every application is defined as a class, having the application name as the class name. An application class has two functions:

### Initializing function

Here, the basic setup of the application is defined. This includes the following elements:

- `base_delta` (integer >= 1) denotes the number of discrete time steps that are grouped together, to render the dynamical system fully actuated. For example, if the dimension of the state is 6, and the dimension of the control input is 3, then it will be at least `base_delta = 2`.
- `adaptive` (dictionary) contains the information of the adaptive measurement scheme, stored in 2 different keys. `adaptive['rates']` (list of integers) defines the adaptive rates; `adaptive['target_points']` can be used to define a different set of target points for the adaptive rate, than for the base rate.
- `control['limits'] ['uMin'] / ['uMax']` (list of integers) are the control authority limits. It is given as a list, and every entry reflects a dimension of the state.
- `partition['nrPerDim']` (list of integers) is the number of regions defined in every dimension of the state space. Note that the partition has a region centred at the origin when odd values are given.
- `partition['width']` (list of floats) is the width of the regions in every dimension of the state space.
- `partition['origin']` (list of floats) denotes the origin of the partitioned region of the state space. 
- `targets['nrPerDim']` (either `'auto'` or a list of integers) is the number of target points defined in every dimension of the state. When `'auto'` is given, the number of target points equals the number of regions of the partition.
- `targets['domain']` (either `'auto'` or a list of integers) is the domain over which target points are defined. When `'auto'` is given, the domain is set equal to that of the state space partition.
- `spec['goal']` (dictionary) is the dictionary describing the goal region. The function `defSpecBlock()` defines these blocks of goal regions, by creating slices in the state space partition in given dimensions.
- `spec['critical']` (dictionary) is the dictionary describing the goal region The function `defSpecBlock()` defines these blocks of critical regions, by creating slices in the state space partition in given dimensions.
- `endTime` (integer > 0) is the finite time horizon over which the reach-avoid problem is solved.

### SetModel function

In the `setModel` function, the linear dynamical system is defined in the following form (see the submitted paper for details): 
$$
\mathbf{x}_{k+1} = A \mathbf{x}_k + B \mathbf{u}_k + \mathbf{q}_k + \mathbf{w}_k,
$$
where:

- `A` is an n x n matrix.
- `B` is an n x p matrix.
- `Q` is a n x 1 column vector that reflects the additive deterministic disturbance (q-term in the equation above).

For some models, the model definition is given in non-discretized form, i.e.
$$
\dot{\mathbf{x}}(t) = A_c\mathbf{x}(t) + B_c\mathbf{u}(t) + \mathbf{q}_c(t) + \mathbf{w}_c(t),
$$
where `tau ` (float) is the time discretization step size, subscript c indicates that these matrices and vectors differ from the ones above. If a continuous-time dynamical model is given, it is discretized using one of two methods:

- Using a forward Euler method.
- Using a Gears discretization method.

------

