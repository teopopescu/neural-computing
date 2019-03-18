%% Clear everything
clear all
close all
%% Create train and test data
[train_num,test_num,train_cat,test_cat,odata] = splitdata('original_car_data.csv','HoldOut',0.2)
%% check unique rows 
height(unique(odata))
%% Exploratory Data Analysis
% in order to do a parrallel coordinate plot, first we need to convert the independent variables to
% numeric equivalents
indv=[double(odata.buying) double(odata.maint) double(odata.doors) double(odata.persons) double(odata.lug_boot) double(odata.safety)] ;
group=odata.acceptability;
labels={'buying','maint','doors','persons','lug_boot','safety'};
parallelcoords(indv,'Group',group,'Labels',labels,'LineWidth',2);
% We can see from the graph that the variables that seem to drive the
% unacceptability are a low number of people and low safety, high mantenance and buying. The variables
% that seem to drive the v good acceptability are low buying price, high
% persons, high lugage boot and high safety. 
%% Means by Group
numeric_table=[array2table(indv) array2table(group)];
VarNames={'buying','maint','doors','persons','lug_boot','safety','acceptability'};
numeric_table.Properties.VariableNames = VarNames;
means = grpstats(numeric_table,'acceptability');
indv_means=table2array(means(:,3:8));
labels={'buying','maint','doors','persons','lug_boot','safety'};
labels=labels';
%% spider plot
% % Axes properties 
 axes_interval = 2; 
 axes_precision = 1;  
% % Spider plot 
 spider_plot(indv_means, labels, axes_interval, axes_precision,... 
 'Marker', 'o',... 
 'LineStyle', '-',... 
 'LineWidth', 2,... 
 'MarkerSize', 5); 
% 
% % Title properties 
 title('Average Value per Predictor by Target Class',... 
 'Fontweight', 'bold',... 
 'FontSize', 12); 
% 
% % Legend properties 
legend_values={'unacc','acc','good','vgood'};
legend('show', 'Location', 'southoutside',legend_values); 

%%
figure(1)
for i=1:6
    [tbl,chi2,p,labels]=crosstab(table2array(odata(:,i)),odata.acceptability);
    rowNames = transpose(labels(:,1));
    colNames = transpose(labels(:,2));
    subplot(2,3,i);
    bar(tbl,'stacked')
    xticklabels(rowNames);
    %legend(colNames,'Location','northwestoutside');
    title(odata.Properties.VariableNames(i));
    hold on
end
%% target variable viz
hist(odata.acceptability);

%Naive Bayes code
%% Read the numerical train dataset for Naive Bayes Model Selection
train_set=readtable('training_num80.csv');
%% specify that the last column is ordinal categorical data
categories={'unacc','acc','good','vgood'};
train_set.acceptability=categorical(train_set.acceptability,categories,'Ordinal',true);
%% Examine if any of the variables are highly correlated. 
CM=corrcoef(table2array(train_set(:,1:6)));
% we can see none of the independent variables are highly correlated, so we
% can include them all. 
%% set the priors and the class names
tab=tabulate(train_set.acceptability); %calculate percentages
prior=cell2mat(transpose(tab(:,3)))/100; %turn them into right format
class_names={'unacc','acc','good','vgood'};
%% Hyper parameter optimization for fincnb
% this optimization can only optimize normal or kernel distributions. So we
% are going to pick the the most optimal values from it and compare it with
% a model which uses the multivariate multinomial distributions. 
%One of the variables that we are also going to explore is the number of
%folds
nb_initialize_results_collection_variables;
for f = 5:10
        %split predictors and target variables
        X=train_set(:,1:6);
        Y=train_set(:,7);
        rng(1);
        %set up the optimization function
        Mdl = fitcnb(X,Y,...
            'ClassNames',class_names,...
            'Prior',prior,...
            'OptimizeHyperparameters','all',...
            'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus',...
            'Kfold',f));
        % Get the best hyper parameters
        BestDist=Mdl.ModelParameters.DistributionNames;
        BestWidth=Mdl.ModelParameters.Width;
        BestKernelType=Mdl.ModelParameters.Kernel;        
        %save the results
        num_folds=[num_folds;f];
        best_dist=[best_dist;cellstr(BestDist)];
        best_width=[best_width;num2cell(BestWidth)];
        best_kernel=[best_kernel;cellstr(BestKernelType)];
end
%% Merge the results into 1 table
nb_hyperp_type_conversion;
normal_kernel_best_params=[num_folds best_dist best_width best_kernel];
%% We can see that for each CV type, we get slightly different results:
% Let's add the accuracy and cross entropy to each of the models above
nb_initialize_results_collection_variables;
for m=1:6
    %set up all the parameters
    nb_initialize_fold_results_collection_variables;
    distributions=char(table2cell(normal_kernel_best_params(m,2)));
    w=cell2mat(table2array(normal_kernel_best_params(m,3)));
    ktype=char(table2cell(normal_kernel_best_params(m,4)));
    num_folds = table2array(normal_kernel_best_params(m,1));
    training_err = zeros(num_folds,1);
    %start the CV folds
    rng(1);
    c = cvpartition(train_set.acceptability,'KFold',num_folds);
    startt=cputime;
    for fold = 1:num_folds
            nb_split_data;            
             % Train a model
                Mdl = fitcnb(Xtr,Ytr,...
                    'ClassNames',class_names,...
                    'Prior',prior,...
                    'DistributionNames',distributions,...
                    'Width',w,...
                    'Kernel',ktype);
            nb_predict_evaluate_test;
    end
        endt=cputime;
        nb_cv_performance;   
end 
%% Add the results to the original table
nb_perfm_type_conversion;
normal_kernel_best_params=[normal_kernel_best_params accuracy cross_entropy time];
%% Compare the previous results with the 'mvmn' distribution across the same fold types
nb_initialize_results_collection_variables;
for f=5:10
    nb_initialize_fold_results_collection_variables;
    distributions='mvmn';
    training_err = zeros(f,1);
    %start the CV folds
    rng(1);
    c = cvpartition(train_set.acceptability,'KFold',f);
    startt=cputime;
    for fold = 1:f
        nb_split_data;
         % Train a model
            Mdl = fitcnb(Xtr,Ytr,...
                'ClassNames',class_names,...  
                'Prior',prior,...
                'DistributionNames',distributions);
         nb_predict_evaluate_test;
    end
    endt=cputime;
    nb_cv_performance;
    %Save the rest of parameters results. 
    num_folds=[num_folds;f];
    best_dist=[best_dist;cellstr(distributions)];
    best_kernel=[best_kernel;cellstr('NA')];
    best_width=[best_width;cellstr('NA')];        
end;
%% Save the results for the multinomial distributions
nb_hyperp_type_conversion;
nb_perfm_type_conversion;
multinomial_results=[num_folds best_dist best_width best_kernel accuracy cross_entropy time];
%% merge the two tables together for final model selection
final_nb_models = [normal_kernel_best_params; multinomial_results];
final_nb_models = sortrows(final_nb_models,6,{'descend'});
best_model=final_nb_models(1,:);
best_model;
writetable(final_nb_models,'final_nb_models.csv');
%% Train the model on the entire train set and make final predictions on the test set. 
% Read the numerical test dataset for Naive Bayes Model Selection
test_set=readtable('test_num80.csv');
% specify that the last column is ordinal categorical data
test_set.acceptability=categorical(test_set.acceptability,categories,'Ordinal',true);
%split predictors and target variables
Xtr=table2array(train_set(:,1:6));
Ytr=table2array(train_set(:,7));
Xv=table2array(test_set(:,1:6));
Yv=table2array(test_set(:,7));
%Train the model with kernel parameters
startt=cputime;
Mdl = fitcnb(Xtr,Ytr,...
            'ClassNames',class_names,...
            'Prior',prior,...
            'DistributionNames','kernel',...
            'Width',0.4293,...
            'Kernel','box');  
endt=cputime;
training_time=endt-startt;
% Predict Result on train set
[label,Posterior,Cost] = predict(Mdl,Xtr);
% Evaluate Model on train set
[ac_train_final, ce_train_final]=performance_metrics(Mdl, Xtr,Ytr, Posterior);

% save the results of predictions for test set
predictions=array2table([Ytr label]);
VarNames={'target','predictions'};
predictions.Properties.VariableNames = VarNames;
writetable(predictions,'final_results_train_nb.csv');

% Predict Result on test set
[label,Posterior,Cost] = predict(Mdl,Xv);
% Evaluate Model on test set
[ac_test_final, ce_test_final]=performance_metrics(Mdl, Xv,Yv, Posterior);

%% save the results of predictions for test set
predictions=array2table([Yv label]);
VarNames={'target','predictions'};
predictions.Properties.VariableNames = VarNames;
writetable(predictions,'final_results_nb.csv');

%Random Forest code
    
%% Hyperparameter setup for performing Grid Search
nb_of_trees=[10 20 75 100]; %Number of trees
max_num_splits=[100 200 300]; % Maximal number of decision splits
min_leaf_size =[2 3 4 5]; %Minimum number of observations per tree leaf
num_variables_to_sample=[1 3 6]; %Number of predictors to select at random for each split.

random_forest_initialize_hyperparameters;
for trees = nb_of_trees
    for nb_predictors = num_variables_to_sample
        for leaf_size = min_leaf_size
            for splits = max_num_splits
      
                %Split predictors and target variables
                features=train_set(:,1:6);
                labels=train_set(:,7);
                %Set up the training function
                start_time=cputime;
                Mdl = TreeBagger(trees, features,labels,...
                'Method','classification',...
                'ClassNames', categorical({'unacc','acc','good','vgood'}),...
                'MaxNumSplits', splits, ...
                'NumVariablesToSample', nb_predictors, ...
                'MinLeafSize', leaf_size, ...
                'SplitCriterion', 'deviance', ...
                'OOBPrediction','on');
                end_time=cputime;
                training_time=end_time-start_time;

%%Compute prediction and score matrix containing the probability of each observation originating from the class, computed as the fraction of observations of the class in a tree leaf.
                [Yfit,scores] = predict(Mdl,features);
% Evaluate model on the train set by computing its cross-entropy
                Yv2 = dummyvar(categorical(table2array(labels)));
                %Replace the '0' in posterior probabilities with 0.000001 so
                %that the log conversion is not infinite. 
                Posterior2=scores;
                Posterior2(Posterior2 == 0)=0.00001;
                %Apply the log to the output vector
                log_o=log(Posterior2);
                %Multiply the result from above with the actual target values
                product=log_o.*Yv2;
                %Calculate cross entropy at row level
                row_e=sum(product,2);
                %Calculate the mean cross entropy for the model
                ce_train_rf=mean(row_e);      
                %Save the results into one table
                number_of_trees=[number_of_trees;trees];
                leafs=[leafs;leaf_size];
                number_of_splits=[number_of_splits;splits];
                number_of_predictors = [number_of_predictors;nb_predictors];
                accuracy=[accuracy; (1-oobError(Mdl,...
         'Mode','Ensemble'))];
                ce=[ce;ce_train_rf];
                time=[time;training_time];
            end
        end
    end
end

%% Merge models' results into one table
random_forest_perform_type_conversion;
rf_models = [number_of_trees leafs number_of_splits number_of_predictors accuracy ce time]
rf_models = sortrows(rf_models,6 ,{'descend'});
writetable(rf_models, 'Random_Forest_Models.csv')

%Take the model with highes cross entropy loss and store it separately;
best_rf_model =rf_models(1,:);
writetable(best_rf_model, 'Best_Random_Forest_model.csv')

%Train again the best model and predict on train set, get predicted labels and save on a comparison file
final_training_start_time=cputime;
bestMdl =  TreeBagger(table2array(best_rf_model(1,1)), features,labels,...
    'Method','classification',...
    'ClassNames', categorical({'unacc','acc','good','vgood'}),...
    'MaxNumSplits', table2array(best_rf_model(1,3)), ...
    'NumVariablesToSample', table2array(best_rf_model(1,4)), ...
    'MinLeafSize', table2array(best_rf_model(1,2)), ...
    'OOBPrediction','on');
final_training_end_time=cputime;
final_train_time = final_training_end_time - final_training_start_time;

%Compute final model accuracy on train set
final_model_accuracy=(1-oobError(bestMdl,...
    'Mode','Ensemble'));

% Predict final results (labels and score matrix) on train set
[Yfit,scores] = predict(bestMdl,features);
  
% Save the results of predictions for train set
  predictions = [Yfit labels];
  VarNames = {'train_predictions','target'};
  predictions.Properties.VariableNames = VarNames;
  writetable(predictions,'final_results_train_Random_Forest.csv');

%%Predict Results on the test set
%Split test set into features and labels
test_features = test_set(:,1:6);
test_labels = test_set(:,7);

[test_Yfit,test_scores] = predict(bestMdl,test_features);

final_results = [test_labels test_Yfit];
VarNames = {'target','test_predictions'};
final_results.Properties.VariableNames = VarNames;
writetable(final_results,'Random_Forest_Final_Labels.csv');

%Compute final accuracy
err = error(bestMdl,test_features,test_Yfit);
accuracy_test_final= 1-mean(err);

%% Confusion Matrix between target test labels and predicted test labels
confusion_matrix = confusionmat(categorical(table2array(test_labels)),categorical(test_Yfit));
plotconfusion(categorical(table2array(test_labels)),categorical(test_Yfit));

%% Compute final cross-entropy
test_set.acceptability=categorical(test_set.acceptability,categories,'Ordinal',true);             
Yv3 = dummyvar(test_set.acceptability);
Posterior2=test_scores;
Posterior2(Posterior2 == 0)=0.00001;
log_o=log(Posterior2);
product=log_o.*Yv3;
row_e=sum(product,2);
ce_test_final=mean(row_e);









