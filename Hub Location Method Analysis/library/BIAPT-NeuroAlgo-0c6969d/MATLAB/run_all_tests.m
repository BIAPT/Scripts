%{
    This script purpose is to run all the tests we have in the _tests
    folder with some added explanation of what they do. This should be run
    after having created a new functionality or made significant changes to
    make sur that nothing is broken.
%}

%% Parameters Tests
% Here we are making sure that the parameters we have used before are still
% correct. If they are not correct we will need to change the test and
% update the documentation
runtests('parameters_test');