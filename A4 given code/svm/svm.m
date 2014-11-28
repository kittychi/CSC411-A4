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
    classid{i} = tr_labels==i; 
end

% reorder tr_images into a NxD matrix,
% where N = number of examples
% and   D = number of dimensions (pixesl in this case)

ntr = size(tr_images, 3);
ntest = size(test_images, 3); 

resized_tr_images = double(reshape(tr_images, [h*w, ntr]));

% take out a subset of the training images for validation purposes
perm = randperm(ntr); 
valID = perm(1:300);
test_id = perm(301:ntr);


X = resized_tr_images'; 
X = X(test_id);
validation = resized_tr_images'(valID); 
validation_labels = tr_labels(valID); 

% train svm 
for i=1:1
    models{i} = fitcsvm(X, classid{i}(test_id), 'CrossVal', 'on'); 
    posteriors{i} = fitSVMPosterior(models{i}); 
    [label, score] = predict(posteriors{i}, validation); 
    labels{i} = label; 
    scores{i} = score; 
end; 
