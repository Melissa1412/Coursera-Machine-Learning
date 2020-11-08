function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));  % 25by401

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));  % 10by26

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% set up a vector of labels for later use (to turn y into a logical array)
K = [];
for k = 1:num_labels,
K = [K; k]; % K = [1;2;3;4;5;6;7;8;9;10]
end;


% forward prop
X = [ones(m, 1) X];
a = sigmoid(X*Theta1');
a = [ones(size(a, 1), 1) a];
h = sigmoid(a*Theta2');

% cost function without regularization   
for c = 1:num_labels,
new_y = (y==c);
J = J - 1/m * (sum(new_y.*log(h(:, c))+(1-new_y).*log(1-h(:, c))));
end;

% regularized cost function
regTheta1 = Theta1;
regTheta1(:, 1) = 0;
regTheta2 = Theta2;
regTheta2(:, 1) = 0;
J = J + lambda/(2*m) * (sum(sum(regTheta1.^2)) + sum(sum(regTheta2.^2)));

% back prop
% g'(z) = sigmoidGradient(z)
for t =1:m,

% STEP 1 -- set up a1 and compute activations for the rest of the layers
a1 = [X(t, :)(:)];		% 401by1
z2 = Theta1*a1;		% 25by401 * 401by1 = 25by1
a2 = [1; sigmoid(z2)];	% 26by1
z3 = Theta2*a2;		% 10by26 * 26by1 = 10by1
a3 = sigmoid(z3);		% 10by1

% STEP 2 -- delta3
delta3 = a3 - (K==y(t));	% 10by1

% STEP 3 -- delta2
delta2 = (Theta2' * delta3)(2:end) .* sigmoidGradient(z2); % 25by1, don't want delta20

% STEP 4 -- accumulate gradient
Theta1_grad = Theta1_grad + delta2 * a1';
Theta2_grad = Theta2_grad + delta3 * a2';

end;

% STEP 5 -- obtain regularized gradients
Theta1_grad = 1/m * Theta1_grad + lambda/m .* regTheta1;
Theta2_grad = 1/m * Theta2_grad + lambda/m .* regTheta2;





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
