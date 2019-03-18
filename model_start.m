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

training_table = data_norm(idx(1:round(P*rows)),:) ; 
test = data_norm(idx(round(P*rows)+1:end),:) ;

%% Bootstrapping aggregating (5 bags)

%to get train set dimensions 
[train_rows,train_columns] = size(training_table);

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
-Split into training and test set x
-Fit model x

-Grid search : Justify why you chose some over the others;
    -Learning rate; <-- plot it on a graph for the best model; Learning
    rates: 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1
    -Momentum: 
    -Activation function (sigmoid, softmax, tanh, RELU) 
    -Alpha (Learning rate decay factor)  
    -Training function will stay the same 
    -Size of hidden layers
    -Max number of epochs

Comment on early stopping and minimum training performance;
Comment on batch size and nb of epochs

https://uk.mathworks.com/help/deeplearning/ref/network.html
https://uk.mathworks.com/matlabcentral/answers/310935-in-neural-network-toolbox-how-can-i-can-change-the-values-of-training-parameters

    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],

-Best model selection
-Use best model on test set;

%}
features = training_table{:,1:13};
labels = training_table{:,14};

%First application of the model using default parameters

% Create a Fitting Network
hiddenLayerSize = 10;
model = patternnet(hiddenLayerSize,'traingda');


feat=training_table{:,1:13};
lab=training_table{:,14};
initial_features =transpose(feat);
initial_labels = transpose(lab);
% Train the Network
[model_2,tr] = train(model,features,labels); 

%Prediction
label=predict(model,training_table(val_idx{1},1:13));
target= table2array(training_table(val_idx{1},14));

%Grid search 
initialize_hyperparameters;
learning_rate=[0.001 0.003 0.01 0.03 0.1 0.3 1]; %Learning rate hyperparameters
momentum =[0.5 0.6 0.7 0.9]; %Momentum hyperparameter
number_of_epochs=[10 20 30]; %Number of epochs hyperparameter
batch_size = [32 64 128 256]; %Batch size hyperparameter
% https://towardsdatascience.com/what-are-hyperparameters-and-how-to-tune-the-hyperparameters-in-a-deep-neural-network-d0604917584a
for z= 1:length(train_idx)
    for rate = learning_rate
        for momentum_rate =momentum
            for epoch = number_of_epochs
                %for batch = batch_size
                %check stack overflow neural network training batch size; 
                %Split predictors and target variables
                train_features=training_table(train_idx{z},1:13);
                train_labels=training_table(train_idx{z},14);
                validation_features=training_table(val_idx{z},1:13);
                validation_labels=training_table(val_idx{z},14);
                %Set up the training function
                %trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.
                trainFcn ='traingscg' % Gradient descent with adaptive learning rate backpropagation
                hiddenLayerSize = 10;
                start_time=cputime;
                % Create a Pattern Recognition Network
                net = patternnet(hiddenLayerSize, trainFcn,'crossentropy');
                net.trainParam.epochs=epoch
                net.trainParam.lr=rate
                net.trainParam.lr_inc=1+momentum_rate;
              
                view(net)
                %Set network hyperparameters
               % options = trainingOptions('Momentum',momentum_rate ,...
                %                          'MaxEpochs',epoch ,...
                 %                         'MiniBatchSize',batch,...
                  %                        'LearnRateDropFactor',rate);
                
                  
                % Train the Network
                [net,tr] = training_table(net,train_features,train_labels);
                end_time=cputime;
                training_time=end_time-start_time;

                %%Compute prediction 
                    YPred_train = predict(net,train_features)
                    YPred_validation = predict(net,validation_features);
                
                    %Compute model accuracy  
                  train_ya=table2array(validation_labels);
                  train_yatrain=table2array(training_labels);
                  train_yd=double(train_ya);
                  train_ydtrain=double(train_yatrain);
                  
              
                %Confusion matrices and its values using validatio set to test results                   
                 confusion = confusionmat(train_yd,YPred_validation);
                 %TrueNegative | TruePositive | FalseNegative | FalsePositive
                 TN=confusion(1,1);
                 TP=confusion(2,2);
                 FN=confusion(2,1);
                 FP=confusion(1,2);
                
                  %Confusion matrices and its values using training
                                        %set to test results
                 confusion2 = confusionmat(train_ydtrain,YPred_train);
                 TN2=confusion2(1,1);
                 TP2=confusion2(2,2);
                 FN2=confusion2(2,1);
                 FP2=confusion2(1,2);
                                        
                 %Accuracy
                  Accuracy=(TN+TP)/(TN+TP+FN+FP);
                  AccuracyTrain=(TN2+TP2)/(TN2+TP2+FN2+FP2);

                 
                %Save the results into one table
                learning_rate_list=[learning_rate_list;rate];
                momentum_list=[momentum_list;momentum_rate];
                number_of_epochs_list=[number_of_epochs_list;epoch];
                %batch_size_list = [batch_size_list;batch_size];
                accuracy_list=[accuracy_list; AccuracyTrain];
                cross_entropy_list=[cross_entropy_list;perform(net,train_features,train_labels)];
                time_list=[time_list;training_time];
            end
        end
    end
end

%% Merge models' results into one table
perform_type_conversion;
mlp_models = [learning_rate_list momentum_list number_of_epochs_list accuracy_list cross_entropy_list time_list]

mlp_models = sortrows(rf_models,6 ,{'descend'});
writetable(rf_models, 'Random_Forest_Models.csv')



