%pool = parpool; % Invoke workers
options = statset('UseParallel',true);

cross_val_partition = cvpartition(size(X,1),'KFold',5);
% ROUGH DRAFT OF WHAT TO DO
error = zeros(cross_val_partition.NumTestSets,1);
for i = 1:cross_val_partition.NumTestSets
    disp(strcat("Cross validation: ",string(i)))
    training_index = cross_val_partition.training(i);
    test_index = cross_val_partition.test(i);
    % THIS USE 5 FOLD CROSS VALIDATION FOR OPTIMIZING THE PARAMETERS
    % What we need to do is feed it the training data and optimize the
    % value using that
    Mdl = fitcecoc(X(training_index,:),Y(training_index),'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'),'Options',options)
    [label_test] = predict(Mdl,X(test_index,:))
    %label_test = classify(meas(test_index,:),meas(training_index,:),species(training_index,:));
    % Then we use predict using the test set and check what the error is
    error(i) = sum(label_test == Y(test_index)');
end
cross_val_error = sum(error)/sum(cross_val_partition.TestSize);
% This cross validation error will be used to check for generalization

% IDEA : here is to first use cross validation to select the right
% hyperparameters to minimize the error.
% Then we use cross validation on top of that to check for the
% generalization of the choosen model.