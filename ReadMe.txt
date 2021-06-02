This artifact contains the source code for the following article, presented at
the NFM 2021 Symposium. The full reference to the submitted paper is as 
follows: 

  Badings, T.S., Hartmanns, A., Jansen, N. and Suilen, M. (2020). Balancing 
  Wind and Batteries: Toward Predictive Verification of Smart Grids. NFM 2021.

The files in this repository contain everything that is needed to replicate 
the results presented in the paper. Our simulations ran on a Windows laptop with Intel Core i7-1065G7 CPU (1.3-3.0 GHz) and 16 GB RAM, but using the instructions below, the experiments may be replicated on a virtual machine as well.

The artifact has been tested with the TACAS 21 Artifact Evaluation VM - Ubuntu 
20.04 LTS, which can be downloaded from the link specified below. To execute 
the program with this Virtual Machine, please follow the steps described below.

For any questions, please send an e-mail to Thom Badings: thom.badings@ru.nl

Python version: 3.8.3. 
For a list of the used packages, please see the requirements.txt file.

----------------------------------------
INSTALLATION AND HOW TO USE THE ARTIFACT

Please follow the instructions below if you want to use the artifact within a
virtual machine. If you plan to use the program directly on your own machine,
you may skip steps 1 and 2.

1. DOWNLOAD AND INSTALL VIRTUALBOX
First, download and install the VirtualBox host from the following webpage:

  https://www.virtualbox.org/wiki/Downloads.

The artifact has been tested with VirtualBox 6.1.18 on Windows 10. 

2. DOWNLOAD THE TACAS 21 VIRTUAL MACHINE
We tested the artifact on the TACAS 21 Artifact Evaluation VM, which can be 
downloaded from Zenodo.org:

  https://zenodo.org/record/4041464

Download the 'TACAS 21.ova' file (size 3.6 GB), then open the VirtualBox 
application, and import this file. After that, boot up the virtual machine.

Note that other virtual machines that support Python 3 may work as well, but 
we tested the artifact specifically for this one.

3. DOWNLOAD THE FILES OF THE ARTIFACT
Download and extract the artifact files to a folder on the virtual machine
(or on your own machine, if you are not using a virtual one) with writing 
access (needed to store results).

4. INSTALL PACKAGES
Open a terminal and navigate to the artifact folder. Then, run the following
command to install the required packages:

  $ pip3 install -r requirements.txt

You can open the requirements.txt file if you want to see the list of specific 
packages that are being installed.

5. RUN THE SCRIPT
Run the "RunFile.py" file to execute the program, by typing the command:

  $ python RunFile.py

Follow the instructions shown on the screen, by choosing values for:
  a) the network configuration (option 1, 2, or 3)
  b) the exploration horizon (300, 600, 900 seconds)
  c) the action discretization level (any positive integer value)
  d) the number of Monte Carlo iterations (any positive integer value)

Note that appropriate combinations of these values allow the user to recreate
the results described in the paper exactly. 

You may also change the parameter values manually, by changing the code. See 
the description below for a detailed example of how to do this.

6. INSPECT THE RESULTS
All results are stored in the 'output/' folder. In particular, the results
include the following (for every iteration of the Monte Carlo simulation):
  a) Figures of the day-ahead power balance, frequency deviations, ancillary 
     service deployment, and state of charge of batteries (in .pdf format);
  b) An excel file, describing all states, controls, and other results of the
     specific iteration.

In addition, the following results are stored for the whole execution (i.e.
all Monte Carlo iterations combined) of the program:
  a) A .pickle file, containing all Python data (can be loaded easily back to
     Python; see the documentation of the package for more information);
  b) An Excel file for the overall results over all iterations, containing the
     optimal objectives, number of states and actions, and more.

Every result figure or file in the output folder contains a timestamp and
information about the specific scenario parameters in its filename.

----------------------------------------
RECREATING THE RESULTS OF THE PAPER

If you want to replicate the results presented in the paper, you may roughly 
follow the following steps.

All results presented can be recreated without any manual modification. As 
such, it is sufficient to simply run the program, and select the appropriate 
options for the parameters shown on the screen. Result figures are recreated
automatically (due to the nature of the Monte Carlo simulation, note that 
actual figures may deviate from the paper). For the statistical analysis (of 
run times, model sizes, etc.), some basic data manipulation in Excel is needed
(e.g., to compute the confidence intervals of every case).

----------------------------------------
MANUAL USER SETTINGS

In addition to the built-in scenario choices, the user may modify the
RunFile.py file for further customization of the execution of the program. The
values of the basic parameters can be hardcoded in the following lines:

Line 62.  Power system network configuration (value of 1, 2, or 3)
Line 73.  Exploration horizon (value of 300, 600, or 900 seconds)
Line 80.  Action discretization level (any positive integer value)
Line 86.  Number of Monte Carlo iterations (any positive integer value)

Some more examples of parameters that may be interesting for modification are:

Line 99.  Boolean switch whether every iteration is plotted
Line 124. Simulation horizon (24 hours by default)
Line 128-129. Frequency limit value (hard constraint)
Line 138. Optimization criterion (standard reward, or pure reachability prob.)
Line 141. Discount factor used in the MDP (1 by default)
Line 149. Boolean switch for Verbose setting (print more or less output)

The list above is definitely not exhaustive, and many more parameters of other
inputs may be modified or customized. Detailed documentation is included in 
every file of the program (although most general settings can be found in the
RunFile.py file).