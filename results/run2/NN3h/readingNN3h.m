function [train_results, test_results] = readingNN3h(order_matrix);
        % Script for reading all the neural network research data from seperate files generated.
        % Takes input the same matrix which was used to generate the data and returns matrix with as $
        % dimensions as there was layers on neural network.

        % Because in the saved files there is distinction between neural nets which have different la$
        % We need to get rid of them. Lambdas are the last column in order_matrix.


        max_lambda = max(order_matrix(:, 4));
	max_hls3 = max(order_matrix(:, 3));
        max_hls2 = max(order_matrix(:, 2));
        max_hls1 = max(order_matrix(:, 1));

        % Placeholders to take lambda differences out.
        temp1 = zeros(max_lambda, 1);
        temp2 = zeros(max_lambda, 1);

        % Return values.
        train_results = zeros(max_hls1, max_hls2, max_hls3);
        test_results = zeros(max_hls1, max_hls2, max_hls3);


        for hls = 1:max_hls1;
        for hls2 = 1:max_hls2;
	for hls3 = 1:max_hls3;
                for lambda = 1:max_lambda;

                % Load appropiate file
                load(sprintf("NN3h-%d-%d-%d-L%d", hls, hls2, hls3, lambda))

                % Save the data.
                temp1(lambda) = trainerror;
                temp2(lambda) = testerror;

                endfor

                % Get the maximum out of all lambdas and save it into results
                train_results(hls, hls2, hls3) = max(temp1);
                test_results(hls, hls2, hls3) = max(temp2);

	endfor
        endfor
        endfor

        train_results
        test_results
endfunction
