load digits; 
values = zeros(4, 9);
k=1;
p_ = zeros(2, 10); 
mu_ = zeros(256, 20); 
vary_ = zeros(256, 20); 
final_logProb = zeros(1, 10); 
mean_logProb = zeros(1, 10); 
logProb = zeros(10, 10); 
for randConst=[0.001,0.5,20,60,150,270,420,500,10000]
for i = 1:10
[p3, mu3, vary3, logProbX3] = mogEM(train3, 2, 10, 0.01, 0, randConst, 0);
p_(:, i) = p3;
mu_(:, 2*i-1:2*i) = mu3;
vary_(:, 2*i-1:2*i) = vary3;
final_logProb(i) = logProbX3(10);
mean_logProb(i) = mean(logProbX3);
logProb(:, i) = logProbX3;
end;
index = find(final_logProb == max(final_logProb));
index = index(1); 
visualize_digits(mu_(:, 2*index-1:2*index), sprintf('Mean with randConst = %d', randConst), sprintf('mean3-%d', randConst));
visualize_digits(vary_(:, 2*index-1:2*index), sprintf('Variance with randConst = %d', randConst),sprintf('vary3-%d', randConst));
values(:, k) = [randConst; final_logProb(index); p_(:, index)]; 
k=k+1;
end;