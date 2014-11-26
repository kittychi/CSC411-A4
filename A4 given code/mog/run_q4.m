
load labeled_images.mat;
load public_test_images.mat;
%load hidden_test_images.mat;

h = size(tr_images,1);
w = size(tr_images,2);

if ~exist('hidden_test_images', 'var')
  test_images = public_test_images;
else
  test_images = cat(3, public_test_images, hidden_test_images);
end

% separating training data into the different classes
c1 = find(tr_labels' == 1); tr_im_1 = tr_images(:, :, c1); 
c2 = find(tr_labels' == 2); tr_im_2 = tr_images(:, :, c2); 
c3 = find(tr_labels' == 3); tr_im_3 = tr_images(:, :, c3); 
c4 = find(tr_labels' == 4); tr_im_4 = tr_images(:, :, c4); 
c5 = find(tr_labels' == 5); tr_im_5 = tr_images(:, :, c5); 
c6 = find(tr_labels' == 6); tr_im_6 = tr_images(:, :, c6); 
c7 = find(tr_labels' == 7); tr_im_7 = tr_images(:, :, c7); 

% reshaping training and test images into vectors
ntr = size(tr_images, 3);
ntest = size(test_images, 3);
tr_images = double(reshape(tr_images, [h*w, ntr]));
test_images = double(reshape(test_images, [h*w, ntest]));

c1 = find(tr_labels' == 1); tr_im_1 = tr_images(:, c1); 
c2 = find(tr_labels' == 2); tr_im_2 = tr_images(:, c2); 
c3 = find(tr_labels' == 3); tr_im_3 = tr_images(:, c3); 
c4 = find(tr_labels' == 4); tr_im_4 = tr_images(:, c4); 
c5 = find(tr_labels' == 5); tr_im_5 = tr_images(:, c5); 
c6 = find(tr_labels' == 6); tr_im_6 = tr_images(:, c6); 
c7 = find(tr_labels' == 7); tr_im_7 = tr_images(:, c7); 

numComponent = [3:10 15 20 35];
% numComponent = [3];

% errorTrain = zeros(size(numComponent));
% errorValidation = zeros(size(numComponent));
% errorTest = zeros(size(numComponent));
% % numComponent = [2, 5, 10, 15, 20, 25, 30, 35];
% bestk= zeros(2, 7);
% acc = zeros(size(numComponent, 2)); 
% for i = 1 : size(numComponent, 2)
%     k = numComponent(i);
% % Train a MoG model with K components for each class
% %-------------------- Add your code here --------------------------------
%     [p1, mu1, vary1, logProbX1] = mogEM(tr_im_1, k, 20, 0.01, 0, 60, 1);
%     [p2, mu2, vary2, logProbX2] = mogEM(tr_im_2, k, 20, 0.01, 0, 60, 1);
%     [p3, mu3, vary3, logProbX3] = mogEM(tr_im_3, k, 20, 0.01, 0, 60, 1);
%     [p4, mu4, vary4, logProbX4] = mogEM(tr_im_4, k, 20, 0.01, 0, 60, 1);
%     [p5, mu5, vary5, logProbX5] = mogEM(tr_im_5, k, 20, 0.01, 0, 60, 1);
%     [p6, mu6, vary6, logProbX6] = mogEM(tr_im_6, k, 20, 0.01, 0, 60, 1);
%     [p7, mu7, vary7, logProbX7] = mogEM(tr_im_7, k, 20, 0.01, 0, 60, 1);
%     
% % Caculate the probability P(d=1|x) and P(d=2|x), 
% % classify examples, and compute the error rate
% % Hints: you may want to use mogLogProb function
% %-------------------- Add your code here --------------------------------
%     
%     mogtestLogProb1 = mogLogProb(p1,mu1,vary1,tr_images);
%     mogtestLogProb2 = mogLogProb(p2,mu2,vary2,tr_images);
%     mogtestLogProb3 = mogLogProb(p3,mu3,vary3,tr_images); 
%     mogtestLogProb4 = mogLogProb(p4,mu4,vary4,tr_images);
%     mogtestLogProb5 = mogLogProb(p5,mu5,vary5,tr_images); 
%     mogtestLogProb6 = mogLogProb(p6,mu6,vary6,tr_images);
%     mogtestLogProb7 = mogLogProb(p7,mu7,vary7,tr_images); 
%     
%     testLogProb = [mogtestLogProb1; mogtestLogProb2; mogtestLogProb3; mogtestLogProb4;
%         mogtestLogProb5; mogtestLogProb6; mogtestLogProb7];
%     
%     c1_pred = testLogProb(:, c1); [~, c1_] = max(c1_pred); class1acc= length(find(c1_ == 1))/size(c1, 2); 
%     c2_pred = testLogProb(:, c2); [~, c2_] = max(c2_pred); class2acc= length(find(c2_ == 1))/size(c2, 2); 
%     c3_pred = testLogProb(:, c3); [~, c3_] = max(c3_pred); class3acc= length(find(c3_ == 1))/size(c3, 2); 
%     c4_pred = testLogProb(:, c4); [~, c4_] = max(c4_pred); class4acc= length(find(c4_ == 1))/size(c4, 2); 
%     c5_pred = testLogProb(:, c5); [~, c5_] = max(c5_pred); class5acc= length(find(c5_ == 1))/size(c5, 2); 
%     c6_pred = testLogProb(:, c6); [~, c6_] = max(c6_pred); class6acc= length(find(c6_ == 1))/size(c6, 2); 
%     c7_pred = testLogProb(:, c7); [~, c7_] = max(c7_pred); class7acc= length(find(c7_ == 1))/size(c7, 2); 
%     
%     if (class1acc > bestk(2, 1)) bestk(:, 1) = [k; class1acc]; end;
%     if (class2acc > bestk(2, 2)) bestk(:, 2) = [k; class2acc]; end;
%     if (class3acc > bestk(2, 3)) bestk(:, 3) = [k; class3acc]; end;
%     if (class4acc > bestk(2, 4)) bestk(:, 4) = [k; class4acc]; end;
%     if (class5acc > bestk(2, 5)) bestk(:, 5) = [k; class5acc]; end;
%     if (class6acc > bestk(2, 6)) bestk(:, 6) = [k; class6acc]; end;
%     if (class7acc > bestk(2, 7)) bestk(:, 7) = [k; class7acc]; end;
% 
%     [maxprob, classification] = max(testLogProb); 
%     misses = find(tr_labels ~= classification'); 
%     matches = find(tr_labels' == classification); 
%     
%     acc(i) = length(matches)/ntr;
%     fprintf('EM with K=%d for all class resulted in %.4f accuracy\n', k, acc(i));
% %     errorTest(1, i)  = size(find(tr_labels ~= classification'), 2); 
% %     
% %     mogValidLogProb2 = mogLogProb(p2,mu2,vary2,inputs_valid);
% %     mogValidLogProb3 = mogLogProb(p3,mu3,vary3,inputs_valid); 
% %     
% %     valid_class = mogValidLogProb2 < mogValidLogProb3; 
% %     errorValidation(1, i)  = size(find(target_valid ~= valid_class), 2); 
% %     
% %     mogTrainLogProb2 = mogLogProb(p2,mu2,vary2,inputs_train);
% %     mogTrainLogProb3 = mogLogProb(p3,mu3,vary3,inputs_train); 
% %     
% %     train_class = mogTrainLogProb2 < mogTrainLogProb3; 
% %     errorTrain(1, i)  = size(find(target_train ~= train_class), 2); 
% end

[maxacc, bestK] = max(acc);
fprintf('overall best K is %d.\n', bestK);
fprintf('Running class 1 with K = %d. \n', bestk(1, 1)); 
fprintf('Running class 2 with K = %d. \n', bestk(1, 2)); 
fprintf('Running class 3 with K = %d. \n', bestk(1, 3)); 
fprintf('Running class 4 with K = %d. \n', bestk(1, 4)); 
fprintf('Running class 5 with K = %d. \n', bestk(1, 5)); 
fprintf('Running class 6 with K = %d. \n', bestk(1, 6)); 
fprintf('Running class 7 with K = %d. \n', bestk(1, 7)); 

[p1, mu1, vary1, logProbX1] = mogEM(tr_im_1, bestk(1, 1), 20, 0.01, 0, 60, 1);
[p2, mu2, vary2, logProbX2] = mogEM(tr_im_2, bestk(1, 2), 20, 0.01, 0, 60, 1);
[p3, mu3, vary3, logProbX3] = mogEM(tr_im_3, bestk(1, 3), 20, 0.01, 0, 60, 1);
[p4, mu4, vary4, logProbX4] = mogEM(tr_im_4, bestk(1, 4), 20, 0.01, 0, 60, 1);
[p5, mu5, vary5, logProbX5] = mogEM(tr_im_5, bestk(1, 5), 20, 0.01, 0, 60, 1);
[p6, mu6, vary6, logProbX6] = mogEM(tr_im_6, bestk(1, 6), 20, 0.01, 0, 60, 1);
[p7, mu7, vary7, logProbX7] = mogEM(tr_im_7, bestk(1, 7), 20, 0.01, 0, 60, 1);

mogtestLogProb1 = mogLogProb(p1,mu1,vary1,test_images);
mogtestLogProb2 = mogLogProb(p2,mu2,vary2,test_images);
mogtestLogProb3 = mogLogProb(p3,mu3,vary3,test_images); 
mogtestLogProb4 = mogLogProb(p4,mu4,vary4,test_images);
mogtestLogProb5 = mogLogProb(p5,mu5,vary5,test_images); 
mogtestLogProb6 = mogLogProb(p6,mu6,vary6,test_images);
mogtestLogProb7 = mogLogProb(p7,mu7,vary7,test_images); 

testLogProb = [mogtestLogProb1; mogtestLogProb2; mogtestLogProb3; mogtestLogProb4;
        mogtestLogProb5; mogtestLogProb6; mogtestLogProb7];
    
[~, prediction] = max(testLogProb); 

% Fill in the test labels with 0 if necessary
if (length(prediction) < 1253)
  prediction = [prediction zeros(1,1253-length(prediction))];
end


% Print the predictions to file
fprintf('writing the output to prediction.csv\n');
fid = fopen('prediction.csv', 'w');
fprintf(fid,'Id,Prediction\n');
for i=1:length(prediction)
  fprintf(fid, '%d,%d\n', i, prediction(i));
end
fclose(fid);