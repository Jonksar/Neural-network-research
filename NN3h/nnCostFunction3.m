function [J grad] = nnCostFunction3(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   hidden_layer_size2, ...
                                   hidden_layer_size3, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a three hidden layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

% Useful variables
ils = input_layer_size;
hls = hidden_layer_size;
hls2 = hidden_layer_size2;
hls3 = hidden_layer_size3;
ols = num_labels; % output layer size
m = size(X, 1);

sizeT1 = (ils +1) * hls;
sizeT2 = (hls +1) * hls2;
sizeT3 = (hls2 +1) * hls3;
sizeT4 = (hls3 +1) * ols;

% Unrolling parameters
Theta1 = reshape(nn_params(1:sizeT1), ...
                 hls, (ils + 1));

Theta2 = reshape(nn_params((sizeT1 + 1):(sizeT1+sizeT2)), ...
                 hls2, (hls + 1));

Theta3 = reshape(nn_params((1 + sizeT1 + sizeT2):(sizeT1 + sizeT2 + sizeT3)), ...
                 hls3, (hls2 + 1));

Theta4 = reshape(nn_params((1 + sizeT1 + sizeT2 + sizeT3):(sizeT1 + sizeT2 + sizeT3 + sizeT4)), ...
                 ols, (hls3 + 1));

% return the following variables
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));
Theta4_grad = zeros(size(Theta4));
s = 0;

% feedforward and backpropagation part, one example at the time.
for i=1:m,
    x_i = [1 X(i,:)]';

    y_i = zeros(num_labels, 1);
    y_i(y(i)) = 1;

    z_2 = Theta1 * x_i;
    a_2 = [1; sigmoid(z_2)];

    z_3 = Theta2 * a_2;
    a_3 = [1; sigmoid(z_3)];

    z_4 = Theta3 * a_3;
    a_4 = [1; sigmoid(z_4)];

    z_5 = Theta4 * a_4;
    h = sigmoid(z_5);

    d_5 = h - y_i;
    d_4 = Theta4' * d_5;
    d_4 = d_4(2:end) .* sigmoidGradient(z_4);
    d_3 = Theta3' * d_4;
    d_3 = d_3(2:end) .* sigmoidGradient(z_3);
    d_2 = Theta2' * d_3;
    d_2 = d_2(2:end) .* sigmoidGradient(z_2);

    Theta4_grad = Theta4_grad + (d_5 * a_4');
    Theta3_grad = Theta3_grad + (d_4 * a_3');
    Theta2_grad = Theta2_grad + (d_3 * a_2');
    Theta1_grad = Theta1_grad + (d_2 * x_i');

    s = s + sum(-y_i .* log(h) .- (1.-y_i) .* log(1.-h));
end

% exclude bias unit
t_1 = Theta1(:,2:end);
t_2 = Theta2(:,2:end);
t_3 = Theta3(:,2:end);
t_4 = Theta4(:,2:end);

% Regularize parameters
Theta1_grad = [Theta1_grad(:,1)/m ((Theta1_grad(:,2:end)/m) + (lambda/m) * t_1)];
Theta2_grad = [Theta2_grad(:,1)/m ((Theta2_grad(:,2:end)/m) + (lambda/m) * t_2)];
Theta3_grad = [Theta3_grad(:,1)/m ((Theta3_grad(:,2:end)/m) + (lambda/m) * t_3)];
Theta4_grad = [Theta4_grad(:,1)/m ((Theta4_grad(:,2:end)/m) + (lambda/m) * t_4)];

% Regulare cost function
t_1 = sum(sum(t_1.^2));
t_2 = sum(sum(t_2.^2));
t_3 = sum(sum(t_3.^2));
t_4 = sum(sum(t_4.^2));
r = (lambda/(2*m))*(t_1 + t_2 + t_3 + t_4);

% Cost for returning.
J = (1/m) * s + r;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:); Theta3_grad(:); Theta4_grad(:)];


end