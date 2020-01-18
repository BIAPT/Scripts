%{
    Yacine Mahdid 2020-01-18
    This script will run all experiments to go from raw .set data to
    publishable figures. It may take a while to run or break in the middle
    if something is wrong with the analysis/data. The script will output
    checkpoint explaining at which point it or which point failed. The
    experiments can then be run manually one by one starting from the
    failing point.
%}

disp("Running experiment 14: generate wpli or dpli matrices:");
try
   experiment_14_generate_w_or_d_pli_matrix 
catch exception
    error(strcat("experiment 14 failed or needs to be changed: ", exception.message));
end

disp("Running experiment 15a: generate the motifs from the dpli matrices");
try 
   experiment_15a_generate_motif
catch exception
    error(strcat("experiment 15a failed or needs to be changed: ", exception.message));
end

disp("Running experiment 15b: generate power spectra from the raw data");
try
   experiment_15b_generate_power_spectra 
catch exception
    error(strcat("experiment 15b failed or needs to be changed: ", exception.message));
end

disp("Running experiment 15c: generate network properties from the wpli matrices");
try
   experiment_15c_generate_network_properties 
catch exception
    error(strcat("experiment 15c failed or needs to be changed: ", exception.message));
end

disp("Running experiment 16: generate figures from the motif/power spectra and network properties");
try
   experiment_16_generate_figures 
catch exception
    error(strcat("experiment 16 failed or needs to be changed: ", exception.message));
end