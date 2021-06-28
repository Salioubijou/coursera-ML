function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%X = [ones(m, 1) X]; % Training examples with bias
hxtheta_y = X * theta - y;
theta_2_n = theta(2:end, :);
% My cost function
J += (hxtheta_y' * hxtheta_y + ...
	lambda * theta_2_n' * theta_2_n) / (2 * m);
% My gradient
grad += [X(:, 1)' * hxtheta_y; X(:, 2:end)' * hxtheta_y + ...
       lambda * theta_2_n] / m;	



% =========================================================================

grad = grad(:);

end
