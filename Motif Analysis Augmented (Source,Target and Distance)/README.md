# Paper: Brain Network Motifs are Markers of Loss and Recovery of Consciousness
This is the code for the first motif analysis paper and it's subsequent review.
It contains analysis for **wpli/dpli**, **motif analysis**, **graph theory network analysis** and **power analysis** of participant from **MDFA** study.

## Structure of the Code
The code is structured in the following manner:
- **exploratory_experiments** folder : contains experiment that lead to the current working code.
- **utils** folder: contain helper functions that are used throughout the analysis
- **experiments_\*.m** at the root level: these are the main script that will produce the reported output.
- **run_all_analysis.m** at the root level: this script will run all the relevant experiments to generate the reported output.
- **settings.txt** at the root level: this text file contains the configuration for the analysis that will be used throughout the experiments.
- **library** folder: this folder contain all the external libraries that are needed to run the experiment. This is mainly the **NeuroAlgo** library. This library is currently in development and does change a lot over time. For this reason one of the release is included and must be used to constantly generated the right output and to prevent the code to break over time.

The code is separated into 5 main experiments instead of being combined into one big script. The rational behind this is that some part of the experiment takes a long time to run. If something goes wrong in the middle of the full output generation we only need to restart this experiment instead of starting again from the first one. The **run_all_analysis.m** is there to run all the experiment automatically if we are sure that nothing is wrong with the output.

## Requirements
- **MATLAB 2018a** and above
- **NeuroAlgo 0.0.1** (can be download from the BIAPT repository if needed)
- **MDFA raw data** (see below)

## How to Use the Code
1. Edit the **settings.txt** input path. The input path needs to contain the MDFA raw data and have this structure.
    - input_folder
        - MDFA03
            - MDFA03_BASELINE.fdt
            - MDFA03_BASELINE.set
            - MDFA03_EMF5.fdt
            - MDFA03_EMF.set
            - MDFA03_EML5.fdt
            - MDFA03_EML5.set
            - MDFA03_EML10.fdt
            - MDFA03_EML10.set
            - MDFA03_EML30.fdt
            - MDFA03_EML30.set
            - MDFA03_IF5.fdt
            - MDFA03_IF5.set
            - MDFA03_RECOVERY.fdt
            - MDFA03_RECOVERY.set
        - MDFA05
            - etc.
        - etc
2. Edit the settings.txt output path. This folder only needs to exist and can be located anywhere on your computer or on an external disk. The resulting output path will be populated like this:
    - motif_analysis_(current_date):
        - wpli
            - MDFA03
            - MDFA05
            - etc.
        - dpli
            - MDFA03
            - MDFA05
            - etc.
        - motif
            - MDFA03
            - MDFA05
            - etc.
        - network
            - MDFA03
            - MDFA05
            - etc.
        - power
            - MDFA03
            - MDFA05
            - etc.
        - figure
            - MDFA03
            - MDFA05
            - etc.
3. Open MATLAB, add the current folder to the path and run `run_all_analysis` and wait for the output to be populated. If you want to run each experiments one by one do `experiment_*` and wait for one experiment to be done.

## Modifying the Code
If you are running a new experiment for a new project it might be a good idea to create a new folder in the `Script` repository and take whatever you need from this folder. If the code needs to be modified for review, please modify the structure, put the relevant experiments into exploratory_experiments and edit this README.md to reflect the changes.

## Something is Wrong?
Go to the issues tab and open an issue explaining what is the problem.

### Authors
- Danielle Nadin
- Yacine Mahdid