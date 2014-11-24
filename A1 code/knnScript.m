%% Clear workspace.
clear all;
close all;

%% Load data.
load mnist_train;
load mnist_valid;
load mnist_test;

labels = valid_targets;
total = size(valid_targets,1);

% first row of index is used to keep track of the k to run
% second row of index is to keep track of the percentage of correct guesses
% from the model
index=[1 3 5 7 9; 0 0 0 0 0];
for k = 1:5
    % runs the k nearest neighbours with values 1, 3, 5, 7, 9 for k
    %trained = run_knn(index(1,k), train_inputs, train_targets, valid_inputs);
    tested= run_knn(index(1, k), test_inputs, test_targets, valid_inputs); 
    
    % find out how many valid inputs are correct based on the model
    %totalcorrect = size(find(trained==labels), 1);
    totalcorrect = size(find(tested==labels), 1); 
    % store the percentage correct in index
    index(2, k) = (totalcorrect/total);
end

% plotting the results
index(2,:)
plot(index(1,:), index(2,:));
xlabel('value of k');
ylabel('classification rate on the validation set');