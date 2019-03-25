%% Data split
%data loading
data =readtable('clean.csv');


%train and test sets
[rows,columns] = size(data);
% 85% of data will be used for training
P = 0.85 ;
% Random split using index
idx = randperm(rows);

training_table = data(idx(1:round(P*rows)),:) ; 
test = data(idx(round(P*rows)+1:end),:) ;

%% Bootstrapping aggregating (5 bags)

%to get train set dimensions 
[train_rows,train_columns] = size(training_table);

%Only the index will be stored, this makes the code scalable to large datasets
%reducing memory usage.

%Loop to create 5 training and validation datasets sampling with replacement
for i=1:5
    
%Five vectors of size N= 80%observations where each n is a random positive number between 1 and 223
    train_idx{i} = randi(round(train_rows*0.80),1,round(train_rows*0.80));
    
%Five vectors of size N= 20%observations where each n is a random positive number between 224 and 279  
    val_idx{i} = round(train_rows*0.80) + randi(round(train_rows*0.20),1,round(train_rows*0.20));
    
    
end

%randi() allows to sample with replacement since we can have the same random number multiple times 


%% MLP model First application of the model using default parameters
%%
% Create a Fitting Network
hiddenLayerSize = 10;
trainFcn = 'traingdx';
net = patternnet(hiddenLayerSize,trainFcn,'crossentropy');

feat=table2array(training_table(train_idx{1},1:13));
lab=table2array(training_table(train_idx{1},14));

% Train the Network
net = train(net,feat',lab'); 

% prediction
validation=(training_table(val_idx{1},1:13));
validation=table2array(validation);
new_label=net(validation');

% The output is the probability of being 1, for this reason when the percentage 
%is above 50% it will be translated to 1, 0 toherwise... 
new_label(1,(new_label(1,:))< 0.5)= 0 ;
new_label(1,(new_label(1,:))>= 0.5)= 1 ;

target= table2array(training_table(val_idx{1},14));
confusion = confusionmat(target,new_label);

%TrueNegative | TruePositive | FalseNegative | FalsePositive
TN=confusion(2,2);
TP=confusion(1,1);
FN=confusion(2,1);
FP=confusion(1,2);

%Accuracy
first_NN_accuracy=(TN+TP)/(TN+TP+FN+FP)

%% Grid search 
%Training is happening on training set, prediction on validation set and accuracy and cross entropy are 
%computed for each validation set

%parameters initialization
learning_rate=[];
momentum=[];
neurons_number=[];
accuracy_indicator=[];
cross_entropy=[];
time=[];
bag=[];

% A first grid search has been performed using 5,10 and 15 epochs, since at
% 5 epochs we already found convergence we will use 5 epochs in combination with
%other parameters to find out which is the best model
epochs= 10;

learning_rate_list=[ 0.003 0.01 0.03 0.1 0.3 1]; %Learning rate hyperparameters
momentum_list =[ 0.2 0.5 0.7 0.9 ]; %Momentum hyperparameter
number_of_neurons=[5 10 15 20]; %Number of epochs hyperparameter

for z= 1:length(train_idx)
    for rate = learning_rate_list
        for momentum_rate =momentum_list
            for neurons = number_of_neurons

                %Split predictors and target variables
                train_features=training_table(train_idx{z},1:13);
                train_labels=training_table(train_idx{z},14);
                validation_features=training_table(val_idx{z},1:13);
                validation_labels=training_table(val_idx{z},14);

                %Transformation into array 
                train_feat = table2array(train_features)
                train_lab = table2array(train_labels)
                valid_feat = table2array(validation_features)
                valid_lab = table2array(validation_labels)

                %Set up the training function
                trainFcn ='traingdx' % Gradient descent with adaptive learning rate backpropagation

                %set up number of neurons
                hiddenLayerSize = neurons;

                %time initialization
                start_time=cputime;

                %Create a Pattern Recognition Network
                net = patternnet(hiddenLayerSize, trainFcn,'crossentropy');
                net.trainParam.epochs=epochs;
                net.trainParam.lr=rate;


                net.trainParam.mc = momentum_rate;

                % Train the Network
                [net,tr] = train(net,train_feat',train_lab');
                end_time=cputime;
                training_time=end_time-start_time;


                %%Compute prediction for validation set
                YPred_validation = net(valid_feat');
                cross_entropy_validation = perform(net,valid_lab',YPred_validation);


                % As before output transformation
                %YPred_validation=YPred_validation';

                YPred_validation(1,(YPred_validation(1,:))< 0.5)= 0;
                YPred_validation(1,(YPred_validation(1,:))>= 0.5)= 1;

                %Confusion matrices and its values using validation set to test results
                confusion = confusionmat(valid_lab,YPred_validation');
                %TrueNegative | TruePositive | FalseNegative | FalsePositive

                %CHANGE THIS
                TP=confusion(1,1);
                TN=confusion(2,2);
                FN=confusion(2,1);
                FP=confusion(1,2);

                accuracy=(TN+TP)/(TN+TP+FN+FP);


                %Save the results into one table
                learning_rate=[learning_rate;rate];
                momentum=[momentum;momentum_rate];
                neurons_number=[neurons_number;neurons];
                accuracy_indicator=[accuracy_indicator; accuracy];
                %accuracy_validation_list=[accuracy_validation_list;accuracy_validation];
                %cross_entropy_list=[cross_entropy_list;1-cross_entropy_train];
                cross_entropy=[cross_entropy;1-cross_entropy_validation];
                %validation_list =[validation_list;1-cross_entropy_validation];
                time=[time;training_time];
                bag=[bag;z];

            end
        end
    end
end

%% Merge models' results into one table
%perform_type_conversion;
mlp_models_results = [learning_rate momentum neurons_number accuracy_indicator cross_entropy time]

mlp_models_results = sortrows(mlp_models_results,5,{'descend'});
%writetable(mlp_models, 'MLP_models.csv');

%Take the model with highest accuracy and store it separately;
best_mlp_model_results =mlp_models_results(1,:);
%writetable(best_mlp_model, 'Best_MLP_model.csv')

%% Training again the best model and predict on train set, getting predicted labels and saving on a comparison file

%Set up the training function
trainFcn ='traingdx' % Gradient descent with adaptive learning rate backpropagation
hiddenLayerSize = best_mlp_model_results(1,3);
final_training_start_time=cputime;

%Create a Pattern Recognition Network
net = patternnet(hiddenLayerSize, trainFcn,'crossentropy');
net.trainParam.lr=best_mlp_model_results(1,1)
net.trainParam.mc = best_mlp_model_results(1,2)
net.trainParam.epochs=5;

%Prepare training and test set
%Split test set into features and labels
training_features=training_table(:,1:13);
training_features=table2array(training_features);
training_labels=training_table(:,14);
training_labels=table2array(training_labels);
test_features = table2array(test(:,1:13));
test_labels =table2array(test(:,14));


%ACTUAL TRAINING HAPPENS HERE
[net,tr] = train(net,training_features',training_labels');
end_time=cputime;
training_time=end_time-start_time;

%% Predict Results on the test set
pred_test = net(test_features');
cross_entropy_test = perform(net,test_labels',pred_test);
predicted_classes_test = vec2ind(pred_test);

final_results = [array2table(test_labels) array2table(predicted_classes_test')];
VarNames = {'target','test_predictions'};
final_results.Properties.VariableNames = VarNames;
writetable(final_results,'MLP_Final_Labels.csv');

%% Compute final accuracy
final_confusion = confusionmat(test_labels,predicted_classes_test');
% Confusion Matrix between target test labels and predicted test labels

%TrueNegative | TruePositive | FalseNegative | FalsePositive
final_TP=final_confusion(1,1);
final_TN=final_confusion(2,2);
final_FN=final_confusion(2,1);
final_FP=final_confusion(1,2);
final_accuracy=(final_TN+final_TP)/(final_TN+final_TP+final_FN+final_FP);


%% Plot confusion matrix between target test labels and predicted test labels
plotconfusion(test_labels.',predicted_classes_test)




