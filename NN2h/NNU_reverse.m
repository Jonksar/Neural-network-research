lambdam = [0.01, 0.1, 1, 10, 100];
trainerror = [];
testerror = [];
temp = zeros(1, numel(lambdam));
threshold = 1e-3;
max_hls1 = 10;
max_hls2 = 10;
m = size(X, 1);

for i = 1:m
  if (y(i) == 0)
  y(i) = 10;
  endif
endfor


for A = fliplr(max_hls1/2:max_hls1)
for B = fliplr(max_hls2/2:max_hls2)
for L = 1:numel(lambdam)

%% Setup the parameters you will use for this exercise
input_layer_size  = 784;  % 28 * 28 Input Images of Digits
hidden_layer_size = A;   % 1st layer
hidden_layer_size2 = B;  % 2nd layer
num_labels = 10;          % 10 labels, from 1 to 10
                          % (note that we have mapped "0" to label 10)
lambda = lambdam(L);

pred = [];
test_pred = [];
should_stop = 0;

fprintf('\nInitializing Neural Network Parameters ...\n')
sizeT1 = (input_layer_size + 1) * hidden_layer_size;
sizeT2 = (hidden_layer_size + 1) * hidden_layer_size2;
sizeT3 = (hidden_layer_size2 + 1) * num_labels;

nn_params = zeros(1, sizeT1 + sizeT2 + sizeT3);

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, hidden_layer_size2);
initial_Theta3 = randInitializeWeights(hidden_layer_size2, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:); initial_Theta2(:); initial_Theta3(:)];
fprintf('\nFinished initializing...\n')
fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 20);

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction2(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   hidden_layer_size2, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

    % Reshape nn_params back to Theta1, Theta2, Theta3
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

pred = predict(Theta1, Theta2, Theta3, X);
test_pred = predict(Theta1, Theta2, Theta3, Xtest);

temp(L, 1) = mean(double(pred == y));
temp(L, 2) = mean(double(test_pred == ytest));

i = 0;
while (cost(1) - cost(numel(cost)) > threshold);
    i += 1;

    [nn_params, cost] = fmincg(costFunction, nn_params, options);

    % Reshape nn_params back to Theta1, Theta2, Theta3
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


    pred = predict(Theta1, Theta2, Theta3, X);
    test_pred = predict(Theta1, Theta2, Theta3, Xtest);


    temp(L, 1) = mean(double(pred == y));
    temp(L, 2) = mean(double(test_pred == ytest));


    fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
    fprintf('\nTest Set Accuracy: %f\n', mean(double(test_pred == ytest)) * 100);
    fprintf('\nTraining iteration: %i, Lambda iteration: %i Hidden layer size: %i 2nd Hls %i\n', i, L, A, B);

endwhile

%    plot(trainerror, 'r');
%    hold on;
%    plot(testerror, 'b')
%    hold off;


    if L == numel(lambdam)
      trainerror = [trainerror, temp(:, 1)];
      testerror = [testerror, temp(:, 2)];
      save dataNN2h_2_reverse traindata testdata	
    endif
    
endfor
endfor
endfor


