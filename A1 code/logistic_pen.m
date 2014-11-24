function [f, df, y] = logistic_pen(weights, data, targets, hyperparameters)
% Calculate log likelihood and derivatives with respect to weights.
%
% Note: N is the number of examples and 
%       M is the number of features per example.
%
% Inputs:
% 	weights:    (M+1) x 1 vector of weights, where the last element
%               corresponds to bias (intercepts).
% 	data:       N x M data matrix where each row corresponds 
%               to one data point.
%	  targets:    N x 1 vector of targets class probabilities.
%   hyperparameters: The hyperparameter structure
%
% Outputs:
%	f:             The scalar error value.
%	df:            (M+1) x 1 vector of derivatives of error w.r.t. weights.
% y:             N x 1 vector of probabilities. This is the output of the classifier.
%

%TODO: finish this function
    numData = size(data, 1); % num rows (N) = 20
    lamda = hyperparameters.weight_regularization;
    
    y = logistic_predict(weights, data);
    
    data = [data ones(numData,1)];
    
    z = data*weights;
    exp_z = exp(-z); 
    f = sum(log(1+exp_z)) + sum((1-targets)'*z) + lamda/2*sum(weights.^2);
    df = ((1-targets)'*data - (1-y)'*data)' + weights*lamda;
end
