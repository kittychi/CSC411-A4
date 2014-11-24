%% To run this program:
%%   First run initBp
%%   Then repeatedly call runBp until convergence.

train_CE_list = zeros(1, num_epochs);
valid_CE_list = zeros(1, num_epochs);
train_inc_list = zeros(1, num_epochs);
valid_inc_list = zeros(1, num_epochs);

start_epoch = total_epochs + 1;


num_train_cases = size(inputs_train, 2);
num_valid_cases = size(inputs_valid, 2);

for epoch = 1:num_epochs
  % Fprop
  h_input = W1' * inputs_train + repmat(b1, 1, num_train_cases);  % Input to hidden layer.
  h_output = 1 ./ (1 + exp(-h_input));  % Output of hidden layer.
  logit = W2' * h_output + repmat(b2, 1, num_train_cases);  % Input to output layer.
  prediction = 1 ./ (1 + exp(-logit));  % Output prediction.

  % Compute cross entropy
  train_CE = -mean(mean(target_train .* log(prediction) + (1 - target_train) .* log(1 - prediction)));
  
  % You can consider the output correct if the 
  % correct label is given a higher probability than the incorrect label
  % You should then count up the total number of examples that are classi?ed incorrectly 
  % according to this criterion for training and validation respectively,  
  % and maintain this statistic at the end of each epoch

  prediction_ = (1 - target_train - prediction);
  numIncorrect_train = nnz(find(prediction_>-0.5&prediction_<0.5));
  
  % Compute deriv
  dEbydlogit = prediction - target_train;

  % Backprop
  dEbydh_output = W2 * dEbydlogit;
  dEbydh_input = dEbydh_output .* h_output .* (1 - h_output) ;

  % Gradients for weights and biases.
  dEbydW2 = h_output * dEbydlogit';
  dEbydb2 = sum(dEbydlogit, 2);
  dEbydW1 = inputs_train * dEbydh_input';
  dEbydb1 = sum(dEbydh_input, 2);

  %%%%% Update the weights at the end of the epoch %%%%%%
  dW1 = momentum * dW1 - (eps / num_train_cases) * dEbydW1;
  dW2 = momentum * dW2 - (eps / num_train_cases) * dEbydW2;
  db1 = momentum * db1 - (eps / num_train_cases) * dEbydb1;
  db2 = momentum * db2 - (eps / num_train_cases) * dEbydb2;

  W1 = W1 + dW1;
  W2 = W2 + dW2;
  b1 = b1 + db1;
  b2 = b2 + db2;

  %%%%% Test network's performance on the valid patterns %%%%%
  h_input = W1' * inputs_valid + repmat(b1, 1, num_valid_cases);  % Input to hidden layer.
  h_output = 1 ./ (1 + exp(-h_input));  % Output of hidden layer.
  logit = W2' * h_output + repmat(b2, 1, num_valid_cases);  % Input to output layer.
  prediction = 1 ./ (1 + exp(-logit));  % Output prediction.
  valid_CE = -mean(mean(target_valid .* log(prediction) + (1 - target_valid) .* log(1 - prediction)));
  
  prediction_ = (1 - target_valid - prediction);
  numIncorrect_valid = nnz(find(prediction_>-0.5&prediction_<0.5));
  
  %%%%%% Print out summary statistics at the end of the epoch %%%%%
  total_epochs = total_epochs + 1;
  if total_epochs == 1
      start_error = train_CE;
  end
  train_CE_list(1, epoch) = train_CE;
  valid_CE_list(1, epoch) = valid_CE;
  train_inc_list(1, epoch) = numIncorrect_train;
  valid_inc_list(1, epoch) = numIncorrect_valid;
  
  fprintf(1,'%d  Train CE=%f, Train Incorrect=%f, Valid CE=%f, Valid Incorrect=%f\n',...
            total_epochs, train_CE, numIncorrect_train, valid_CE, numIncorrect_valid);
end

clf; 
if total_epochs > min_epochs_per_plot
  epochs = [1 : total_epochs];
end

%%%%%%%%% Plot the learning curve for the training set patterns %%%%%%%%%
train_errors(1, start_epoch : total_epochs) = train_CE_list;
valid_errors(1, start_epoch : total_epochs) = valid_CE_list;
train_inc(1, start_epoch : total_epochs) = train_inc_list;
valid_inc(1, start_epoch : total_epochs) = valid_inc_list;
  hold on, ...
  plot(epochs(1, 1 : total_epochs), train_errors(1, 1 : total_epochs), 'b'),...
  plot(epochs(1, 1 : total_epochs), valid_errors(1, 1 : total_epochs), 'g'),...
%   plot(epochs(1, 1 : total_epochs), train_inc(1, 1 : total_epochs), 'k'),...
%   plot(epochs(1, 1 : total_epochs), valid_inc(1, 1 : total_epochs), 'r'),...
  legend('Train', 'Test'),...
%   legend('Train incorrect', 'Test incorrect'),...
  title('Cross Entropy vs Epoch'), ...
  xlabel('Epoch'), ...
  ylabel('Cross Entropy');
