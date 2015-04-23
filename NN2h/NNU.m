function [testdata, traindata] = NNU(X, Xtest, y, ytest, input)

    load ordermatrix;

    % Variables we will be using
    lambdam = [0.01, 0.1, 1, 10, 100];
    trainerror = zeros(numel(lambdam), 1);
    testerror = zeros(numel(lambdam), 1);
    threshold = 2e-3;
    lambda = lambdam(order_matrix(input, 3));

    % Defines what structure are we using
    input_layer_size  = 784;  % 28 * 28 Input Images of Digits
    num_labels = 10;          % 10 labels, from 1 to 10
    hidden_layer_size = order_matrix(input, 1);
    hidden_layer_size2 = order_matrix(input, 2);

    % Useful variables
    m = size(X, 1);
    sizeT1 = (input_layer_size + 1) * hidden_layer_size;
    sizeT2 = (hidden_layer_size + 1) * hidden_layer_size2;
    sizeT3 = (hidden_layer_size2 + 1) * num_labels;

    % Shorthand names
    ils = input_layer_size;
    hls = hidden_layer_size;
    hls2 = hidden_layer_size2;
    ols = num_labels; % output layer size

    pred = [];
    test_pred = [];

    printf("Input %i defines network with Hidden layer size: %i 2nd Hls: %i Lambda: %i\n",
                                            input, hidden_layer_size, hidden_layer_size2, lambda)

    fprintf('\nGenerating initial Neural Network Parameters ...\n')


    nn_params = zeros(1, sizeT1 + sizeT2 + sizeT3);

    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    initial_Theta2 = randInitializeWeights(hidden_layer_size, hidden_layer_size2);
    initial_Theta3 = randInitializeWeights(hidden_layer_size2, num_labels);
    % Unroll parameters
    initial_nn_params = [initial_Theta1(:); initial_Theta2(:); initial_Theta3(:)];
    fprintf('\nTraining Neural Network...\n')

    options = optimset('MaxIter', 120);

    % Create "short hand" for the cost function to be minimized
    costFunction = @(p) nnCostFunction2(p, ...
                                       input_layer_size, ...
                                       hidden_layer_size, ...
                                       hidden_layer_size2, ...
                                       num_labels, X, y, lambda);

    % Now, costFunction is a function that takes in only one argument (the
    % neural network parameters) train first with the initial parameters,
    % save results to nn_params.
    [cost, dummy] = costFunction(initial_nn_params);

    nn_params = initial_nn_params;

    i = 0;
    do;

    i += 1;
    prev_cost = cost;
    [nn_params, cost] = fmincg(costFunction, nn_params, options);


    % Unrolling parameters
    Theta1 = reshape(nn_params(1:sizeT1), ...
                     hls, (ils + 1));

    Theta2 = reshape(nn_params((sizeT1 + 1):(sizeT1+sizeT2)), ...
                     hls2, (hls + 1));

    Theta3 = reshape(nn_params((1 + sizeT1 + sizeT2):(sizeT1 + sizeT2 + sizeT3)), ...
                     ols, (hls2 + 1));

    pred = predict(Theta1, Theta2, Theta3, X);
    test_pred = predict(Theta1, Theta2, Theta3, Xtest);

    trainerror = mean(double(pred == y));
    testerror = mean(double(test_pred == ytest));
    fprintf('\nTraining Set Accuracy: %f', trainerror * 100);
    fprintf('\nTest Set Accuracy: %f\n', testerror * 100);

    until(prev_cost - cost) < threshold || i < 20;

    printf("Saving file %s\n", sprintf("NN2h-%d-%d-L%d", hls, hls2, order_matrix(input, 3)))
    save(sprintf("NN2h-%d-%d-L%d", hls, hls2, order_matrix(input, 3)), "trainerror", "testerror")

endfunction
