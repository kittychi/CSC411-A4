load digits;
x = [train2, train3];
%-------------------- Add your code here --------------------------------
% Train a MoG model with 20 components on all 600 training vectors
% with both original initialization and your kmeans initialization. 

final_logProb = zeros(2, 5); 
for i=1:5
    [p2, mu3, vary3, logProbX3] = mogEM(x, 20, 20, 0.01, 0, 60, 1);
    final_logProb(1, i) = logProbX3(20); 
end;

for i=1:5
    [p3, mu3, vary3, logProbX3] = mogEM(x, 20, 20, 0.01, 0, 60, 0);
    final_logProb(2, i) = logProbX3(20); 
end;