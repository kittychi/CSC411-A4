function logistic_regression_template(learningRate, numIterations, regularization, trainingSet, penalize)


%% Load data.
if strcmp(trainingSet,'test') == 1
    load mnist_test;
    inputs = test_inputs; 
    targets = test_targets;
    figureTitle='mnist\_test';
elseif strcmp(trainingSet,'small') == 1
    load mnist_train_small; 
    inputs = train_inputs_small; 
    targets = train_targets_small; 
    figureTitle='mnist\_train\_small';
else
    load mnist_train; 
    inputs = train_inputs; 
    targets = train_targets; 
    figureTitle='mnist\_train';
end
load mnist_valid;

%% TODO: Initialize hyperparameters.
% Learning rate
hyperparameters.learning_rate = eval(learningRate);
% Weight regularization parameter
hyperparameters.weight_regularization = eval(regularization)
% Number of iterations
hyperparameters.num_iterations = eval(numIterations)
% Logistics regression weights
% TODO: Set random weights.
% weights = randn(size(inputs, 2)+1, 1);
weights = ones(size(inputs, 2)+1, 1)*0.4;
% weights = 1./(2.*(1:size(inputs,2)+1))'

%% Verify that your logistic function produces the right gradient, diff should be very close to 0
% this creates small random data with 20 examples and 10 dimensions and checks the gradient on
% that data.
nexamples = 20;
ndimensions = 10;
diff = checkgrad('logistic', ...
	             randn((ndimensions + 1), 1), ...   % weights
                 0.001,...                          % perturbation
                 randn(nexamples, ndimensions), ... % data        
                 rand(nexamples, 1), ...            % targets
                 hyperparameters)                        % other hyperparameters

N = size(inputs, 1);
%% Begin learning with gradient descent.
valid=zeros(hyperparameters.num_iterations,1); 
train=zeros(hyperparameters.num_iterations,1);
for t = 1:hyperparameters.num_iterations
	% Find the negative log likelihood and derivative w.r.t. weights.
    
    if eval(penalize) 
	[f, df, predictions] = logistic_pen(weights, ...
                                           inputs, ...
                                           targets, ...
                                           hyperparameters);
    else
        	[f, df, predictions] = logistic(weights, ...
                                           inputs, ...
                                           targets, ...
                                           hyperparameters);
    end
    
    [cross_entropy_train, frac_correct_train] = evaluate(targets, predictions);
    
	% Find the fraction of correctly classified validation examples.
    if eval(penalize)
	[temp, temp2, frac_correct_valid] = logistic_pen(weights, ...
                                                 valid_inputs, ...
                                                 valid_targets, ...
                                                 hyperparameters);
    else
        [temp, temp2, frac_correct_valid] = logistic(weights, ...
                                                 valid_inputs, ...
                                                 valid_targets, ...
                                                 hyperparameters);
    end

    if isnan(f) || isinf(f)
		error('nan/inf error');
	end

	%% Update parameters.
	weights = weights - hyperparameters.learning_rate .* df / N;

    predictions_valid = logistic_predict(weights, valid_inputs);
    [cross_entropy_valid, frac_correct_valid] = evaluate(valid_targets, predictions_valid);
    
    %% Plot some points
    valid(t) = cross_entropy_valid;
    train(t) = cross_entropy_train;
	%% Print some stats.
	fprintf(1, 'ITERATION:%4i   NLOGL:%4.2f TRAIN CE %.6f TRAIN FRAC:%2.2f VALID_CE %.6f VALID FRAC:%2.2f\n',...
			t, f/N, cross_entropy_train, frac_correct_train*100, cross_entropy_valid, frac_correct_valid*100);

end
index=(1:hyperparameters.num_iterations);
figure;
plot(index, valid, '-', index, train, '--');
xlabel('iterations');
ylabel('cross entropy');
title(figureTitle);
end