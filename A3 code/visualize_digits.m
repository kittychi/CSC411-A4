function visualize_digits(data_matrix, titlestr, filename)
% Visualize digit images for examples in the data matrix.
%
% data_matrix should be a n_dimensions x n_examples matrix, each column is one
% example.
% 
% This is intended only to visualize a small number (say < 10) of digits.
%
if filename
    h=figure('visible', 'off');
else 
    figure; 
end;
 
n_examples = size(data_matrix, 2);
for i = 1 : n_examples
    subplot(2, n_examples/2, i);
    imshow(reshape(data_matrix(:,i), [16,16]), []);
end
if titlestr
    suptitle(titlestr); 
end;

if filename
    saveas(h,filename,'jpg');
end;