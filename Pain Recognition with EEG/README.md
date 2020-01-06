# Pain Recognition with EEG
In this project, we are analyzing 24 channels dry EEG data coming from the Shrinner Hospital in two patient population: Healthy and participant with MSK.

## Analysis Flow
In this folder there are at the time of writing about 6 experiments. The one that matters are the experiment 5,6 and 3,4 in that order. Merging these experiments into two would be a great idea and is set as a new issue. Here is how to run the experiment:
0) Make sure you have the cleaned data sitting in your computer.


1) Run experiment_5_gathering_all_participant: Which will calculate all the features for all the participant and store it in a result folder. Right now the data is still incomplete and a bit messy for some participant, so there will be some refactoring to do.
2) Run experiment_6_averaging_participant_all_channels: Which will run through each of the participants result and create an average participant result for each of the metric studied. This will create a (M | H)EAVG.mat which will be used for the plotting.
3) Run experiment_3_plotting_participants: To make all the plots except the wPLI/dPLI plot. Save these figures for latter.
4) Finally run experiment_4_reordering_matrix: Which will take the unordered wPLI/dPLI matrix and will input an order of frontal to posterior `{'Fp1','Fp2','F3','Fz','F4','F7','F8','C3','Cz','C4','T3','T4','T5','T6','P3','Pz','P4','O1','O2'}`.
