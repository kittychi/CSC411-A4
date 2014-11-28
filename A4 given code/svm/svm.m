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



% create the indicators for each of the 7 classes
for i=1:7 
    classid{i} = ~(tr_labels==i); 
end

% reorder tr_images into a NxD matrix,
% where N = number of examples
% and   D = number of dimensions (pixesl in this case)

ntr = size(tr_images, 3);
ntest = size(test_images, 3); 

resized_tr_images = double(reshape(tr_images, [h*w, ntr]));
resized_test_images = double(reshape(test_images, [h*w, ntr]));
resized_test_images = resized_test_images';

% take out a subset of the training images for validation purposes
perm = randperm(ntr); 
valID = perm(1:300);
test_id = perm(301:ntr);


X = resized_tr_images(:, test_id); 
X = X';
X_labels = tr_labels(test_id); 
validation = resized_tr_images(:, valID);
validation = validation'; 
validation_labels = tr_labels(valID); 

% % train svm 1-vs-all for all classes
% % have not tested this yet
%
% for i=1:7
%     tic;
%     % gaussian svms
% %     models{i} = fitcsvm(X, classid{i}(test_id), 'KernelFunction', 'rbf',...
% %         'BoxConstraint', Inf); 
% 
%     % linear svms
%     models{i} = fitcsvm(X, classid{i}(test_id)); 
%     posteriors{i} = fitSVMPosterior(models{i}); 
%     [label, score] = predict(posteriors{i}, validation); 
%     labels{i} = label; 
%     scores{i} = score; 
%     toc;
% end; 

% train svm 1 vs 1 for all possible pairs
c = combnk(1:7, 2);
for i=1:length(c)
% for i=1:1
    c1 = c(i, 1); 
    c2 = c(i, 2);    
    % generating the test and validation set for this pair of class
    X_id = (X_labels==c1 | X_labels==c2);
    X_ = X(X_id, :);
    X_labels_ = X_labels(X_id);
    
    v_id = (validation_labels==c1 | validation_labels == c2); 
    validation_ = validation(v_id, :); 
    validation_labels_ = validation_labels(v_id); 
    
    tic;
    % gaussian svms
%     models{i} = fitcsvm(X, classid{i}(test_id), 'KernelFunction', 'rbf',...
%         'BoxConstraint', Inf); 

    % linear svms
    models{i} = fitcsvm(X_, X_labels_); 
    posteriors{i} = fitSVMPosterior(models{i}); 
    [label, score] = predict(posteriors{i}, validation_); 
 
    mislabelled = find(label - validation_labels_ ~= 0);
    
    fprintf('(%d, %d) mislabelled %d out of %d\n', c1, c2,...
        length(mislabelled), length(label));
    
    mislabel{i} = mislabelled;
    labels{i} = label; 
    scores{i} = score; 
    toc;
end; 

predictions = zeros(size(resized_test_images, 1), length(posteriors)); 
for i = 1:length(posteriors)
    [prediction, score] = predict(posteriors{i}, resized_test_images); 
    predictions(:, i) = prediction; 
end; 

p = mode(predictions, 2); 

% Fill in the test labels with 0 if necessary
if (length(p) < 1253)
  p = [p; zeros(1253-length(prediction), 1)];
end

fid = fopen('prediction_svm.csv', 'w');
fprintf(fid,'Id,Prediction\n');
for i=1:length(p)
  fprintf(fid, '%d,%d\n', i, p(i));
end
fclose(fid);