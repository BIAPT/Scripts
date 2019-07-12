# Amplitude Envelope Correlation and weighted Phase Lag Index Comparison

This script was developed to compare the decoding performance of source localized Amplitude Envelope Correlation (AEC) and weighted Phase Lag Index (wPLI) in anesthethic induced unconsciousness electroencephalogric (EEG) state in healthy individual (n = 0)

## Definition and Assumption
- [TODO: define AEC]
- [TODO: define wPLI]

## How it works?
There are two main scripts that need to be run in order:
- preprocessing.m, which is a MATLAB script to preprocess the data in order to re-organize the already computed source localized AEC and wPLI and to calculate normalized features set. At the end of the program the data will be located in the /data folder. There will be a X.mat matrix which corresponds to the feature set for every data point, a Y.mat matrix which corresponds to the label of each of these data point and a I.mat matrix which corresponds to the the identity of the participants.
- machine_learning.py, which is a python script that will use the X,Y and I matrices to run various machine learning analysis. 

## How to run?
- To install the required package you need to have python 3.7 > and MATLAB installed.
- Once installed run `pip install requirements.txt` to install the required python packages

## Future
Transform all of this into a python script, there is no real need for MATLAB here.

## Author
Yacine Mahdid 2019-07-06.

If something is unclear raise an issue.
