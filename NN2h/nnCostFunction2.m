function [J grad] = nnCostFunction2(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   hidden_layer_size2, ...
                                   num_labels, ...
                                   X, y, lambda)
%   NNCOSTFUNCTION Implements the neural network cost function for a two hidden layer
%   neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.

	% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
	% for our 2 layer neural network

	ils = input_layer_size;
	hls = hidden_layer_size;
	hls2 = hidden_layer_size2;
	ols = num_labels; % output layer size

	sizeT1 = (ils +1) * hls;
	sizeT2 = (hls +1 ) * hls2;
	sizeT3 = (hls2 +1) * ols;

	Theta1 = reshape(nn_params(1:sizeT1), ...
	                 hls, (ils + 1));

	Theta2 = reshape(nn_params((sizeT1 + 1):(sizeT1+sizeT2)), ...
	                 hls2, (hls + 1));

	Theta3 = reshape(nn_params((1 + sizeT1 + sizeT2):(sizeT1 + sizeT2 + sizeT3)), ...
	                 ols, (hls2 + 1));


	% Setup some useful variables
	m = size(X, 1);

	% Return values
	J = 0;
	Theta1_grad = zeros(size(Theta1));
	Theta2_grad = zeros(size(Theta2));
	Theta3_grad = zeros(size(Theta3));

	s = 0;
	
	% Loops thorugh all the training examples
	for i=1:m,
		% Feedforward
		x_i = [1 X(i,:)]';

		y_i = zeros(num_labels, 1);
		y_i(y(i)) = 1;

		z_2 = Theta1 * x_i;
		a_2 = [1; sigmoid(z_2)];

		z_3 = Theta2 * a_2;
		a_3 = [1; sigmoid(z_3)];

		z_4 = Theta3 * a_3;
		h = sigmoid(z_4);

		% Delta computation
		d_4 = h - y_i;
		d_3 = Theta3' * d_4;
		d_3 = d_3(2:end) .* sigmoidGradient(z_3);
		d_2 = Theta2' * d_3;
		d_2 = d_2(2:end) .* sigmoidGradient(z_2);

		% Computing gradient without regularisation
		Theta3_grad = Theta3_grad + (d_4 * a_3');
		Theta2_grad = Theta2_grad + (d_3 * a_2');
		Theta1_grad = Theta1_grad + (d_2 * x_i');

		s = s + sum(-y_i .* log(h) .- (1.-y_i) .* log(1.-h));

	end
	
	% Removing constant parameters from regularisation
	t_1 = Theta1(:, 2:end);
	t_2 = Theta2(:, 2:end);
	t_3 = Theta3(:, 2:end);
	
	% Computing regularisation term
	Theta1_grad = [Theta1_grad(:,1)/m ((Theta1_grad(:,2:end)/m) + (lambda/m) * t_1)];
	Theta2_grad = [Theta2_grad(:,1)/m ((Theta2_grad(:,2:end)/m) + (lambda/m) * t_2)];
	Theta3_grad = [Theta3_grad(:,1)/m ((Theta3_grad(:,2:end)/m) + (lambda/m) * t_3)];

	t_1 = sum(sum(t_1.^2));
	t_2 = sum(sum(t_2.^2));
	t_3 = sum(sum(t_3.^2));
	r = (lambda/(2*m))*(t_1 + t_2 + t_3);
	
	% Adding regularisation to the cost
	J = (1/m) * s + r;

	% Unroll gradients for returning
	grad = [Theta1_grad(:) ; Theta2_grad(:); Theta3_grad(:)];


end
