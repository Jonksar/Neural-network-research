#! /usr/bin/octave -qf

# Neural network research bash script
arg_list = argv ();

printf("Training artificial neural network; Given input was %s\n", arg_list{1}); 

load traindata5000;
load testdata400;

if str2num(arg_list{1}) <= 100;
	cd NN2h
	NNU(X, Xtest, y, ytest, str2num(arg_list{1}));
else
	cd NN3h
	NNU(X, Xtest, y, ytest, str2num(arg_list{1}) - 100); 
endif
