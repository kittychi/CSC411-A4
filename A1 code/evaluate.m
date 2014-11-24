function [ce, frac_correct] = evaluate(targets, y)
%    Compute evaluation metrics.
%    Inputs:
%        targets : N x 1 vector of targets.
%        y       : N x 1 vector of probabilities.
%    Outputs:
%        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
%        frac_correct : (scalar) Fraction of inputs classified correctly.

% TODO: Finish this function

    numData = size(targets,1);
    ce = 0;
    correct = 0; 
    for i=1:numData
        % getting the target value
        y_i=targets(i); 

        % get probablities from y
        prob_1i = y(i);
        prob_0i = 1-y(i); 

        % keeping the running sum of the log loss 
        ce = ce - y_i*log(prob_1i) - (1-y_i)*log(prob_0i);
        
        if (round(prob_0i) == 1 && y_i == 0) || (round(prob_1i)==1 && y_i ==1)
            correct = correct + 1; 
        end
    end
    
    frac_correct = correct/numData;
end
