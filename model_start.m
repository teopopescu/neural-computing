%% Data split
%data loading
data =readtable('clean.csv');

%SMOTE
[SMOTE_features,SMOTE_labels]=SMOTE( data{:,1:13},  data{:,14});
SMOTE_features = array2table(SMOTE_features);
SMOTE_labels = array2table(SMOTE_labels);
smote_data = [SMOTE_features SMOTE_labels]
writetable(smote_data, 'SMOTE_clean.csv')
normalization [0,1]
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


-Train and validation set (create function here)  x

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

% https://towardsdatascience.com/what-are-hyperparameters-and-how-to-tune-the-hyperparameters-in-a-deep-neural-network-d0604917584a

%}
features = training_table{:,1:13};
labels = training_table{:,14};

%First application of the model using default parameters
%%
% Create a Fitting Network
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize,'traingda');

feat=training_table{:,1:13};
lab=training_table{:,14};
initial_features =transpose(feat);
initial_labels = transpose(lab);
% Train the Network
[net,tr] = train(net,features',labels'); 

%%What is this line of code doing?
new_label=net(features');
%target= table2array(training_table(val_idx{1},14));

%%
%Grid search 
initialize_hyperparameters;
learning_rate=[0.001 0.003 0.01 0.03 0.1 0.3 1]; %Learning rate hyperparameters
momentum =[0.5 0.6 0.7 0.9]; %Momentum hyperparameter
number_of_epochs=[10 20 30]; %Number of epochs hyperparameter
batch_size = [16 32 64 128]; %Batch size hyperparameter
for z= 1:length(train_idx)
    for rate = learning_rate
        for momentum_rate =momentum
            for epoch = number_of_epochs
                for batch = batch_size
                %Split predictors and target variables
                    train_features=training_table(train_idx{z},1:13);
                    train_labels=training_table(train_idx{z},14);
                    validation_features=training_table(val_idx{z},1:13);
                    validation_labels=training_table(val_idx{z},14);
                
                    %CHECK IF YOU NEED THIS
                    train_feat = table2array(train_features)
                    train_lab = table2array(train_labels)
                    valid_feat = table2array(validation_features)
                    valid_lab = table2array(validation_labels)
                
                    %Set up the training function
                    trainFcn ='trainscg' % Gradient descent with adaptive learning rate backpropagation
                    hiddenLayerSize = 10;
                    start_time=cputime;
                    % Create a Pattern Recognition Network
                    net = patternnet(hiddenLayerSize, trainFcn,'crossentropy');
                    net.trainParam.epochs=epoch
                    net.trainParam.lr=rate
                  
                    net.trainParam.mc = momentum_rate
                    %net.trainParam.lr_inc=1+momentum_rate;
                    
                    % Train the Network
                    %ACTUAL TRAINING HAPPENS HERE
                    [net,tr] = train(net,train_feat',train_lab');
                    end_time=cputime;
                    training_time=end_time-start_time;
                    
                    %How to train a neural net for classification from documentation
                    %net = train(net,train_feat',train_lab');
                    %y = net(train_feat');
                    %perf = perform(net,train_lab',y);
                    %classes = vec2ind(y);

                    %%Compute prediction for train set
                    YPred_train =net(train_feat')
                    cross_entropy_train = perform(net,train_lab',YPred_train);
                    classes_train = vec2ind(YPred_train);
                    
                    %%Compute prediction for validation set
                    YPred_validation = net(valid_feat');
                    cross_entropy_validation = perform(net,valid_lab',YPred_validation);
                    classes_validation = vec2ind(YPred_validation);

                    YPred_train = YPred_train'
                    YPred_validation= YPred_validation'
                
                 %Confusion matrices and its values using training set to test results
                 %confusion = confusionmat(valid_lab,YPred_validation);
                 confusion = confusionmat(train_lab,classes_train');
                 %TrueNegative | TruePositive | FalseNegative | FalsePositive
            
                 %CHANGE THIS
                 TN=confusion(1,1);
                 TP=confusion(2,2);
                 FN=confusion(2,1);
                 FP=confusion(1,2);
                
                  %Confusion matrices and its values using validation set to test results
                 %confusion2 = confusionmat(train_lab,classes');
                 confusion2 = confusionmat(valid_lab,classes_validation');
                 TN2=confusion2(1,1);
                 TP2=confusion2(2,2);
                 FN2=confusion2(2,1);
                 FP2=confusion2(1,2);
                                        
                 %Accuracy
                  accuracy_train=(TN+TP)/(TN+TP+FN+FP);
                  accuracy_validation=(TN2+TP2)/(TN2+TP2+FN2+FP2);

               %Save the results into one table
                learning_rate_list=[learning_rate_list;rate];
                momentum_list=[momentum_list;momentum_rate];
                number_of_epochs_list=[number_of_epochs_list;epoch];
                batch_size_list = [batch_size_list;batch_size];
                accuracy_train_list=[accuracy_train_list; accuracy_train];
                accuracy_validation_list=[accuracy_validation_list;accuracy_validation];
                cross_entropy_list=[cross_entropy_list;1-cross_entropy_train];
                validation_list =[validation_list;1-cross_entropy_validation];
                time_list=[time_list;training_time];
                end
            end
        end
    end
end

%% Merge models' results into one table
perform_type_conversion;
mlp_models = [learning_rate_list momentum_list number_of_epochs_list batch_size_list accuracy_train_list accuracy_validation_list cross_entropy_list validation_list time_list]

mlp_models = sortrows(mlp_models,8 ,{'descend'});
writetable(mlp_models, 'MLP_models.csv')

%%Select best model and run on test set;

%Take the model with highest accuracy and store it separately;

best_mlp_model =mlp_models(1,:);
writetable(best_mlp_model, 'Best_MLP_model.csv')

%Train again the best model and predict on train set, get predicted labels and save on a comparison file
final_training_start_time=cputime;

%Fix confusion matrix on grid search 
%Sort out cross-validation/bootstrap aggregation issue 16:30

%Compute final model accuracy on train set

% Predict final results (labels and score matrix) on train set

% Save the results of predictions for train set

%%Predict Results on the test set
%Split test set into features and labels

%Compute final accuracy

%Confusion Matrix between target test labels and predicted test labels



