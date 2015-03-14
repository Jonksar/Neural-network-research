function [testdata, traindata] = NNU(X, Xtest, y, ytest, input)

    % Defines what structure are we using
    load ordermatrix;
    hidden_layer_size = order_matrix(input, 1);
    hidden_layer_size2 = order_matrix(input, 2);
    hidden_layer_size3 = order_matrix(input, 3);

    % Variables we will be using

    input_layer_size  = 784;  % 28 * 28 Input Images of Digits
    num_labels = 10;          % 10 labels, from 1 to 10
    lambdam = [0.01, 0.1, 1, 10, 100];
    trainerror = zeros(numel(lambdam), 1);
    testerror = zeros(numel(lambdam), 1);
    threshold = 8e-5;
    m = size(X, 1);

    for i = 1:m
        if (y(i) == 0)
        y(i) = 10;
        endif
    endfor


    for L = 1:numel(lambdam)

        lambda = lambdam(L);

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
        fprintf('\nTraining Neural Network...\n')

        options = optimset('MaxIter', 120);

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

        % Reshape nn_params back to Theta1, Theta2, Theta3, Theta4
        ils = input_layer_size;
        hls = hidden_layer_size;
        hls2 = hidden_layer_size2;
        hls3 = hidden_layer_size3;
        ols = num_labels; % output layer size

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

        trainerror(L, 1) = mean(double(pred == y));
        testerror(L, 1) = mean(double(test_pred == ytest));
        fprintf('\nTraining Set Accuracy: %f', trainerror(L, 1) * 100);
        fprintf('\nTest Set Accuracy: %f\n\n', testerror(L, 1) * 100);

        i = 0;
        while (cost(1) - cost(numel(cost)) > threshold);
            i += 1;

            [nn_params, cost] = fmincg(costFunction, nn_params, options);

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

            % Save data
            trainerror(L, 1) = mean(double(pred == y));
            testerror(L, 1) = mean(double(test_pred == ytest));

            fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
            fprintf('\nTest Set Accuracy: %f\n', mean(double(test_pred == ytest)) * 100);
            fprintf('\nTraining iteration: %i, Lambda iteration: %i Hidden layer size: %i 2nd Hls %i 3rd Hls %i\n',
                                                i, L, hidden_layer_size, hidden_layer_size2, hidden_layer_size3);

        endwhile

    endfor

    save(sprintf("NN2h-%d-%d-%d", hls, hls2, hls3), "trainerror", "testerror")

endfunction