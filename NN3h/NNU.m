%% Setup the parameters you will use for this exercise
input_layer_size  = 784;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 1st layer
hidden_layer_size2 = 25;  % 2nd layer
hidden_layer_size3 = 25;  % third layer
num_labels = 10;          % 10 labels, from 1 to 10
                          % (note that we have mapped "0" to label 10)

trainerror = [];
testerror = [];
pred = [];
test_pred = [];
should_stop = 0;
lambda = 1;
threshhold = 1e-5;
fprintf('\nLoading data and initializing ... \n')
fprintf('\nLoading 5000 examples of training data...\n')
%load('traindata5000');
fprintf('\nLoading 400 examples of test data...\n')
%load('testdata400');

m = size(X, 1);

% Converting label 0 to 10.
for i = 1:m
  if (y(i) == 0)
  y(i) = 10;
  endif
endfor

fprintf('\nGenerating initial Neural Network Parameters ...\n')
sizeT1 = (input_layer_size + 1) * hidden_layer_size;
sizeT2 = (hidden_layer_size + 1) * hidden_layer_size2;
sizeT3 = (hidden_layer_size2 + 1) * hidden_layer_size3;
sizeT4 = (hidden_layer_size3 + 1) * num_labels;

nn_params = zeros(1, sizeT1 + sizeT2 + sizeT3 + sizeT4);

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, hidden_layer_size2);
initial_Theta3 = randInitializeWeights(hidden_layer_size2, hidden_layer_size3);
initial_Theta4 = randInitializeWeights(hidden_layer_size3, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:); initial_Theta2(:); initial_Theta3(:); initial_Theta4(:)];
fprintf('\nFinished initializing...\n')
fprintf('\nTraining Neural Network...\n')

% must be larger than 1!
options = optimset('MaxIter', 20);

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction3(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   hidden_layer_size2, ...
                                   hidden_layer_size3, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters) train first with the initial parameters,
% save results to nn_params.

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

    % Reshape nn_params back to Theta1, Theta2, Theta3
    ils = input_layer_size;
    hls = hidden_layer_size;
    hls2 = hidden_layer_size2;
    hls3 = hidden_layer_size3;
    ols = num_labels; % output layer size

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

pred = predict(Theta1, Theta2, Theta3, Theta4, X);
test_pred = predict(Theta1, Theta2, Theta3, Theta4, Xtest);
trainerror(1) = mean(double(pred == y));
testerror(1) = mean(double(test_pred == ytest));
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
fprintf('\nTest Set Accuracy: %f\n', mean(double(test_pred == ytest)) * 100);

i = 0;
while not(should_stop);
    i += 1;
      fprintf('\nDifference in cost: %f\n', cost(1) - cost(2))
      % could be changed to cost(1) - cost(numel(cost)) > bigger_Threshold for longer iteration periods
      if (cost(1) - cost(numel(cost)) > 0.001)

      [nn_params, cost] = fmincg(costFunction, nn_params, options);
      % Useful variables
      ils = input_layer_size;
      hls = hidden_layer_size;
      hls2 = hidden_layer_size2;
      hls3 = hidden_layer_size3;
      ols = num_labels; % output layer size

      % Unrolling parameters, keep in mind that nn_params is still ready to go for another training
      Theta1 = reshape(nn_params(1:sizeT1), ...
                       hls, (ils + 1));

      Theta2 = reshape(nn_params((sizeT1 + 1):(sizeT1+sizeT2)), ...
                       hls2, (hls + 1));

      Theta3 = reshape(nn_params((1 + sizeT1 + sizeT2):(sizeT1 + sizeT2 + sizeT3)), ...
                       hls3, (hls2 + 1));

      Theta4 = reshape(nn_params((1 + sizeT1 + sizeT2 + sizeT3):(sizeT1 + sizeT2 + sizeT3 + sizeT4)), ...
                       ols, (hls3 + 1));

    % Calculate error on given outer iteration
    pred = predict(Theta1, Theta2, Theta3, Theta4, X);
    test_pred = predict(Theta1, Theta2, Theta3, Theta4, Xtest);

    % Save error on given outer iteration
    trainerror(i+1) = mean(double(pred == y));
    testerror(i+1) = mean(double(test_pred == ytest));

    % Print the results
    fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
    fprintf('\nTest Set Accuracy: %f\n', mean(double(test_pred == ytest)) * 100);
    fprintf('\n Outer iteration: %i\n', i)
    else
    should_stop = 1;
  endif
endwhile

    plot(trainerror, 'r');
    hold on;
    plot(testerror, 'b')
    hold off;

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
fprintf('\nTest Set Accuracy: %f\n', mean(double(test_pred == ytest)) * 100);
fprintf('\nFinished.\n');