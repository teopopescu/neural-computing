function [Accuracy,AccuracyTrain,Fscore,FscoreTrain] = SVM_calculations(train,val_idx,train_idx,models,counter,z,i,j)
% Calculation of Accuracy and F-score of training and validation set
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

end

