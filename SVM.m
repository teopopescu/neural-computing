
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
    
  %randi() allows to sample with replacement since we can have the same random number multiple times   
end



%% SVM Model

%First application of the model using default parameters

model =fitcsvm(train(train_idx{1},1:13),train(train_idx{1},14));

%prediction
label=predict(model,train(val_idx{1},1:13));

target= table2array(train(val_idx{1},14));

mdlSVM = fitPosterior(model);
[~,score_svm] = resubPredict(mdlSVM);

[X,Y,T,AUC] = perfcurve(target,label,1);
% plot the ROC AUC <-- add title and description 
plot(X,Y)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification by SVM')

%% GRID SEARCH ALGORITHM, 
% inner loop using various number of predictor, the
% one in middle allows to change number of trees, while the outer one is
% used to have our results with k different cross validated. 

counter=0; % Subsequently used to save all the iterations as table observations( j*i*z ) 
box = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1];
Kernel= ["polynomial","linear","RBF"] 
order = [2, 3, 4, 5]
for z= 1:length(train_idx)
     for i= 1:length(box) 
            for j= 1:length(Kernel)   
                        
                        counter=counter+1;
                        
                        if (Kernel(j) == 'linear') 
                                
                                tic;    
                                models{counter}= fitcsvm(train(train_idx{z},1:13),train(train_idx{z},14),'KernelFunction',Kernel(j),'BoxConstraint', box(i));
                                t(counter)=toc; %save the time initialised to zero at tic, gives us the training time 


                                %varius transformations required to commpute the
                                %confusion matrix. We did twice to compute training and validation results.


                                results_j=predict(models{counter},train(val_idx{z},1:13));
                                results_jtrain=predict(models{counter},train(train_idx{z},1:13));



                                train_ya=table2array(train(val_idx{z},14));
                                train_yatrain=table2array(train(train_idx{z},14));


                                train_yd=double(train_ya);
                                train_ydtrain=double(train_yatrain);

                                %Confusion matrices and its values using validation
                                %set to test results                   
                                confusion = confusionmat(train_yd,results_j);

                                %TrueNegative | TruePositive | FalseNegative | FalsePositive
                                TN=confusion(1,1);
                                TP=confusion(2,2);
                                FN=confusion(2,1);
                                FP=confusion(1,2);

                                %Confusion matrices and its values using training
                                %set to test results
                                confusion2 = confusionmat(train_ydtrain,results_jtrain);
                                TN2=confusion2(1,1);
                                TP2=confusion2(2,2);
                                FN2=confusion2(2,1);
                                FP2=confusion2(1,2);


                                %Accuracy
                                Accuracy=(TN+TP)/(TN+TP+FN+FP);
                                AccuracyTrain=(TN2+TP2)/(TN2+TP2+FN2+FP2);


                                %Caluculation of Fscore
                                precision=TP/(TP+FP);
                                recall= TP/(TP+FN);
                                Fscore=2*precision*recall/(precision+recall);

                                precision2=TP2/(TP2+FP2);
                                recall2= TP2/(TP2+FN2);
                                FscoreTrain=2*precision2*recall2/(precision2+recall2);               


                                table(counter,:)=[z,box(i),Kernel(j),'none',FscoreTrain,Fscore,AccuracyTrain,Accuracy,t(counter)]

                        end
                        
                        if (Kernel(j) == 'RBF')
                                
                                tic;    
                                models{counter}= fitcsvm(train(train_idx{z},1:13),train(train_idx{z},14),'KernelFunction',Kernel(j),'BoxConstraint', box(i));
                                t(counter)=toc; %save the time initialised to zero at tic, gives us the training time 


                                %varius transformations required to commpute the
                                %confusion matrix. We did twice to compute training and validation results.


                                results_j=predict(models{counter},train(val_idx{z},1:13));
                                results_jtrain=predict(models{counter},train(train_idx{z},1:13));



                                train_ya=table2array(train(val_idx{z},14));
                                train_yatrain=table2array(train(train_idx{z},14));


                                train_yd=double(train_ya);
                                train_ydtrain=double(train_yatrain);

                                %Confusion matrices and its values using validation
                                %set to test results                   
                                confusion = confusionmat(train_yd,results_j);

                                %TrueNegative | TruePositive | FalseNegative | FalsePositive
                                TN=confusion(1,1);
                                TP=confusion(2,2);
                                FN=confusion(2,1);
                                FP=confusion(1,2);

                                %Confusion matrices and its values using training
                                %set to test results
                                confusion2 = confusionmat(train_ydtrain,results_jtrain);
                                TN2=confusion2(1,1);
                                TP2=confusion2(2,2);
                                FN2=confusion2(2,1);
                                FP2=confusion2(1,2);


                                %Accuracy
                                Accuracy=(TN+TP)/(TN+TP+FN+FP);
                                AccuracyTrain=(TN2+TP2)/(TN2+TP2+FN2+FP2);


                                %Caluculation of Fscore
                                precision=TP/(TP+FP);
                                recall= TP/(TP+FN);
                                Fscore=2*precision*recall/(precision+recall);

                                precision2=TP2/(TP2+FP2);
                                recall2= TP2/(TP2+FN2);
                                FscoreTrain=2*precision2*recall2/(precision2+recall2);               


                                table(counter,:)=[z,box(i),Kernel(j),'none',FscoreTrain,Fscore,AccuracyTrain,Accuracy,t(counter)]

                        end
                        
                        if (Kernel(j) == 'polynomial') 
                               for x= 1:length(order)
                                        tic;    
                                        models{counter}= fitcsvm(train(train_idx{z},1:13),train(train_idx{z},14),'KernelFunction',Kernel(j),'PolynomialOrder',order(x),'BoxConstraint',box(i));
                                        t(counter)=toc; %save the time initialised to zero at tic, gives us the training time 


                                        %varius transformations required to commpute the
                                        %confusion matrix. We did twice to compute training and validation results.


                                        results_j=predict(models{counter},train(val_idx{z},1:13));
                                        results_jtrain=predict(models{counter},train(train_idx{z},1:13));



                                        train_ya=table2array(train(val_idx{z},14));
                                        train_yatrain=table2array(train(train_idx{z},14));


                                        train_yd=double(train_ya);
                                        train_ydtrain=double(train_yatrain);

                                        %Confusion matrices and its values using validation
                                        %set to test results                   
                                        confusion = confusionmat(train_yd,results_j);

                                        %TrueNegative | TruePositive | FalseNegative | FalsePositive
                                        TN=confusion(1,1);
                                        TP=confusion(2,2);
                                        FN=confusion(2,1);
                                        FP=confusion(1,2);

                                        %Confusion matrices and its values using training
                                        %set to test results
                                        confusion2 = confusionmat(train_ydtrain,results_jtrain);
                                        TN2=confusion2(1,1);
                                        TP2=confusion2(2,2);
                                        FN2=confusion2(2,1);
                                        FP2=confusion2(1,2);


                                        %Accuracy
                                        Accuracy=(TN+TP)/(TN+TP+FN+FP);
                                        AccuracyTrain=(TN2+TP2)/(TN2+TP2+FN2+FP2);


                                        %Caluculation of Fscore
                                        precision=TP/(TP+FP);
                                        recall= TP/(TP+FN);
                                        Fscore=2*precision*recall/(precision+recall);

                                        precision2=TP2/(TP2+FP2);
                                        recall2= TP2/(TP2+FN2);
                                        FscoreTrain=2*precision2*recall2/(precision2+recall2);               


                                        table(counter,:)=[z,box(i),Kernel(j),order(x),FscoreTrain,Fscore,AccuracyTrain,Accuracy,t(counter)]
                                        counter=counter+1; %subsequent index
                               end
                               counter=counter-1; % to avoid to skip an index when out of the loop
                        end
           end  
     end
end

%% Table transformations 

%From array to table of values
table_end=array2table(table);

%Columns have been renamed 
table_end.Properties.VariableNames = {'bag' 'box' 'Kernel' 'order' 'Fscore_train' 'Fscore' 'Accuracy_train' 'Accuracy' 'time'};

%Numeric values transformed
table_end.bag=double(table_end.bag);
table_end.box=double(table_end.box);
table_end.Fscore_train=double(table_end.Fscore_train);
table_end.Fscore=double(table_end.Fscore);
table_end.Accuracy_train=double(table_end.Accuracy_train);
table_end.Accuracy=double(table_end.Accuracy);
table_end.time=double(table_end.time);



%% Best model selection
% the most accurate model is selected
[~,modelidx]=max(double(table(:,8)));  
Accmodel= models{modelidx};

% the model with the best Fscore is selected
[~,modelidx2]=max(double(table(:,6)));
Fscoremodel=models{modelidx2};

% Best accuracy model and its hyperparameters
disp('The model with the best accuracy has : '); 

box_constraint=table(modelidx,2);
str1 = ['Box constraint :',num2str(box_constraint)];
disp(str1);

kernel=table(modelidx,3);
str2 = ['Kernel type : ',num2str(kernel)];
disp(str2);

korder=table(modelidx,4);
str4 = ['Kernel Order : ',num2str(korder)];
disp(str4);

bag=table(modelidx,1);
str3 = ['bag training n : ',num2str(bag)];
disp(str3);


% Best F1-Score model and its hyperparameters
fprintf('\nThe model with the best Fscore has :\n'); 
box_constraint2=table(modelidx2,2);
str1 = ['Box constraint :',num2str(box_constraint2)];
disp(str1);

kernel2=table(modelidx2,3);
str2 = ['Kernel type : ',num2str(kernel2)];
disp(str2);

korder2=table(modelidx2,4);
str4 = ['Kernel Order : ',num2str(korder2)];
disp(str4);

bag2=table(modelidx2,1);
str3 = ['bag training n : ',num2str(bag2)];
disp(str3);

% Since the selected model was too much affected by the particular bag rather than
% the hyperparameters, results will be averaged and the most consistent model will be
% picked up. 

%Being a balanced dataset Fscore and Accuracy are highly correlated, for
%this reasons we will carry on the analysis considering only the accuracy of the model.


%% Averageing of results
ord = [2, 3, 4, 5, "none"];
cnt=0;
 for i= 1:length(box) 
        for j= 1:length(Kernel) 
                  for k= 1:length(ord)
                    cnt=cnt+1  ;
                    avg_time = mean(table2array(table_end((table_end.box==box(i)) & (table_end.Kernel==Kernel(j)) & (table_end.order==ord(k)),9)));  
                    avg_Acc = mean(table2array(table_end((table_end.box==box(i)) & (table_end.Kernel==Kernel(j)) & (table_end.order==ord(k)),8)));  

                    avg_table(cnt,:)=[box(i),Kernel(j),ord(k),avg_Acc,avg_time];

                    
                  end
        end
 end
 
 %the inner for loop creates rows even when kernel= linear and
 %order=2,3,4,; those lines will be removed
 avg_table = rmmissing(avg_table);
 
 %avg_table has 42 observation = 210/5(num of bags), the avg has been
 %correctly computed

%% best hyperparameters of avareaged values (5 bags)

% the most accurate averaged hyperparameters are selected
[~,avg_idx]=max(double(avg_table(:,4)));  

% Best accuracy model and its hyperparameters
disp('The model with the best average accuracy has : '); 

box_constraint=avg_table(avg_idx,1);
str1 = ['Box constraint :',num2str(box_constraint)];
disp(str1);

avg_kernel=avg_table(avg_idx,2);
str2 = ['Kernel type : ',num2str(kernel)];
disp(str2);

avg_korder=avg_table(avg_idx,3);
str3 = ['Kernel Order : ',num2str(avg_korder)];
disp(str3);



%% Best models given best avg hyperparameters
%Here we select the model with the best accuracy among the 5 bags, first we
%find the number of the bag
[~,bag_num]=max(table2array(table_end((table_end.box==double(box_constraint)) & (table_end.Kernel==avg_kernel) & (table_end.order==avg_korder),8))) 

%after we filter it using a boolean array and we got the best model
SVMmodel= models{(table_end.bag==bag_num) & (table_end.box==double(box_constraint)) & (table_end.Kernel==avg_kernel) & (table_end.order==avg_korder)};


%% TEST OF THE MODEL
%prediction
test_results=predict(SVMmodel,test(:,1:13));

%target transformation
test_labels=double(table2array(test(:,14)));

%Confusion matrices and its values using validation
%set to test results                   
confusion = confusionmat(test_labels,test_results);

%TrueNegative | TruePositive | FalseNegative | FalsePositive
TN=confusion(1,1);
TP=confusion(2,2);
FN=confusion(2,1);
FP=confusion(1,2);

%Accuracy
Avg_accuracy=(TN+TP)/(TN+TP+FN+FP);

fprintf('\n SVM model accuracy (averaging 5 bags results and selecting best hyper parameters)  : %.4f ', Avg_accuracy)
