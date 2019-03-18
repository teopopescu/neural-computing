%% Data split
%data loading
data =readtable('clean.csv');

%normalization [0,1]
data_norm =normalize(data,'range');

%train and test sets
[rows,columns] = size(data_norm);
% 85% of data will be used for training
P = 0.85 ;
% Random split using index
idx = randperm(rows);

train = data_norm(idx(1:round(P*rows)),:) ; 
test = data_norm(idx(round(P*rows)+1:end),:) ;

%% Bootstrapping aggregating (5 bags)

%to get train set dimensions 
[train_rows,train_columns] = size(train);

%Only the index will be stored, this makes the code scalable to large datasets
%reducing memory usage.

%Loop to create 5 training and validation datasets sampling with replacement
for i=1:5
    
%Five vectors of size N= 80%observations where each n is a positive number between 1 and 223
    train_idx{i} = randi(round(train_rows*0.80),1,round(train_rows*0.80));
    
%Five vectors of size N= 20%observations where each n is a positive number between 224 and 279  
    val_idx{i} = round(train_rows*0.80) + randi(round(train_rows*0.20),1,round(train_rows*0.20))
    
    
end

%randi() allows to sample with replacement since we can have the same random number multiple times 



%% MLP model


%{
Order of code:
-Data split x
-Bootstrapping aggregating x
-Train and validation set (create function here)  x

-Split into training and test set
-Fit model
-Grid search : Justify why you chose some over the others;
    -Learning rate;
    -Momentum
    -Alpha (Learning rate decay factor) 
    -Training function will stay the same 
    -Size of hidden layers
    -Activation function (sigmoid, softmax, tanh, RELU) 
    -Max number of epochs

Comment on early stopping and minimum training performance;

https://uk.mathworks.com/help/deeplearning/ref/network.html
https://uk.mathworks.com/matlabcentral/answers/310935-in-neural-network-toolbox-how-can-i-can-change-the-values-of-training-parameters

    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],

The batch size is a hyperparameter of gradient descent that controls the number of training samples to work through before the model’s internal parameters are updated.
The number of epochs is a hyperparameter of gradient descent that controls the number of complete passes through the training dataset.

-Best model selection
-Use best model on test set;

Grid search
%}

%First application of the model using default parameters

% Create a Fitting Network
hiddenLayerSize = 10;
model = fitnet(hiddenLayerSize,'traingd');

% Train the Network
[model,tr] = train(model,train(train_idx{1},1:13),train(train_idx{1},14));

%Prediction
label=predict(model,train(val_idx{1},1:13));
target= table2array(train(val_idx{1},14));


%Finish all the code by 8pm; start writing paper 21:30

