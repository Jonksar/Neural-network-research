function p = predict(Theta1, Theta2, Theta3, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

a1 = [ones(m,1), X];
z2 = a1 * Theta1';
a2 = [ones(size(z2,1),1), sigmoid(z2)];
z3 = a2 *Theta2';
a3 = [ones(size(z3,1), 1), sigmoid(z3)];
z4 = a3 * Theta3';
a4 = sigmoid(z4);

[predict_max, index_max] = max(a4, [], 2);


p = index_max;


% =========================================================================


end
