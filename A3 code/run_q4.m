load digits;
[inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test] = load_data();
numComponent = [2, 5, 10, 15, 20, 25, 30, 35];
errorTrain = zeros(size(numComponent));
errorValidation = zeros(size(numComponent));
errorTest = zeros(size(numComponent));
% numComponent = [2, 5, 10, 15, 20, 25, 30, 35];

for i = 1 : size(numComponent, 2)
    K = numComponent(i);
% Train a MoG model with K components for digit 2
%-------------------- Add your code here --------------------------------
    [p2, mu2, vary2, logProbX2] = mogEM(train2, K, 20, 0.01, 0, 60, 1);

% Train a MoG model with K components for digit 3
%-------------------- Add your code here --------------------------------
    [p3, mu3, vary3, logProbX3] = mogEM(train3, K, 20, 0.01, 0, 270, 1);

% Caculate the probability P(d=1|x) and P(d=2|x), 
% classify examples, and compute the error rate
% Hints: you may want to use mogLogProb function
%-------------------- Add your code here --------------------------------
    mogtestLogProb2 = mogLogProb(p2,mu2,vary2,inputs_test);
    mogtestLogProb3 = mogLogProb(p3,mu3,vary3,inputs_test); 
    
    test_class = mogtestLogProb2 < mogtestLogProb3; 
    errorTest(1, i)  = size(find(target_test ~= test_class), 2); 
    
    mogValidLogProb2 = mogLogProb(p2,mu2,vary2,inputs_valid);
    mogValidLogProb3 = mogLogProb(p3,mu3,vary3,inputs_valid); 
    
    valid_class = mogValidLogProb2 < mogValidLogProb3; 
    errorValidation(1, i)  = size(find(target_valid ~= valid_class), 2); 
    
    mogTrainLogProb2 = mogLogProb(p2,mu2,vary2,inputs_train);
    mogTrainLogProb3 = mogLogProb(p3,mu3,vary3,inputs_train); 
    
    train_class = mogTrainLogProb2 < mogTrainLogProb3; 
    errorTrain(1, i)  = size(find(target_train ~= train_class), 2); 
end

% Plot the error rate
%-------------------- Add your code here --------------------------------
figure; 
hold on; 
plot(numComponent,errorTest./size(target_test,2), 'r'); 
plot(numComponent,errorTrain./size(target_train,2), 'b'); 
plot(numComponent,errorValidation./size(target_valid,2), 'g'); 
title('number components versus error count');
xlabel('number components');
ylabel('error count');
legend('Test', 'Train', 'Validation'); 