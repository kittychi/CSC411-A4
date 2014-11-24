% Learn a Naive Bayes classifier on the digit dataset, evaluate its
% performance on training and test sets, then visualize the mean and variance
% for each class.

[inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test] = load_data();

% Add your code here (it should be less than 10 lines)
[log_prior, class_mean, class_var] = train_nb(inputs_train, target_train);

[prediction, accuracy] = test_nb(input_test, target_test, log_prior, class_mean, class_var);

visualize_digits([class_mean class_var]);