function L = margLik_cust_fun(sg,rx,ry,y)
% returns the negative log marginal likelihood and gradient for the
% customised covariance function
%
% INPUTS
%   sg - hyperparameters (signal variance, length scale, noise std)  
%   rx, ry - matrices storing the differences between x- and y-components
%   y - measurements
%
% OUTPUTS
%   L - negative log marginal likelihood   
%   g - gradient
%
try
    %% extract hyperparameters
    sigma_f=sg(1); 
    l=sg(2);
    sigma_n=sg(3);
    
    %% # measurements
    n_obs=length(y)/2;
    
    %% Gram matrix
    K=sigma_f^2*[(1-ry.^2/l^2)/l^2 rx.*ry/l^4
        rx.*ry/l^4 (1-rx.^2/l^2)/l^2].*repmat(exp(-0.5*(rx.^2+ry.^2)/l^2),2,2)+sigma_n^2*eye(2*n_obs);
    K = K+2*abs(min([eig(K); 0]))*eye(size(K)); % increase numerical stability
    
    %% calculate function value and gradient
    alpha = K\y;
    
    L = 0.5*y'*alpha+0.5*sum(log(eig(K)))+n_obs * log(2*pi); % - log ML
catch % case something goes wrong
    L=nan;
end
end