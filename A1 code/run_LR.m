%% Clear workspace.
clear all;
close all;

% regular
logistic_regression_template('1', '70', '0.01', 'normal', '0');
logistic_regression_template('1', '70', '0.01', 'small', '0');
logistic_regression_template('1', '70', '0.01', 'test', '0');

%penalized functions
logistic_regression_template('1', '70', '0.01', 'normal', '1');
logistic_regression_template('1', '70', '0.01', 'small', '1');
logistic_regression_template('1', '70', '0.01', 'test', '1');
