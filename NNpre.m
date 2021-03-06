%% Load tha Data
data = readtable('Data.csv');
%To replace missing values with the nearest one
data2 = fillmissing(data,'nearest');
%% PREPROCESSING
%Shape: 
[noRows, noCols] = size(data2);  


%delete duplicate observations 
data2=unique(data2);


%Target with no disease are labeled as 0, while different degree of gravity
%went from 1 up to 4, we are interested in classify the precence of a
%disease, no matter the gravity.

data2.Target(data2.Target == 4,:)= 1;
data2.Target(data2.Target == 3,:)= 1;
data2.Target(data2.Target == 2,:)= 1;

%Outcome variable: Number of 'yes' and 'no'
disease = sum(data2.Target);
no_disease = noRows - disease;

disease_rate = disease/(noRows)

fprintf('Percentage of disease cases : %f ',disease_rate)

fprintf('\nPercentage of no disease cases : %f ',(1-disease_rate))

% Add visualization of Percentage of disease vs no disease


%% ANALYSIS AND BASIC STATISTICS
% first summary of the data
SumData = summary(data2);


%Group data (y)
G = findgroups(data2.Target);

%Calculate descriptive statistics of numerical variables

MeanAge = splitapply(@mean,data2.Age,G);
StdAge = splitapply(@std,data2.Age,G);
SkewAge = splitapply(@skewness,data2.Age,G);
MaxAge = splitapply(@min,data2.Age,G);
MinAge = splitapply(@min,data2.Age,G);

MeanChol = splitapply(@mean,data2.chol,G);
StdChol = splitapply(@std,data2.chol,G);
SkewChol = splitapply(@skewness,data2.chol,G);
MaxChol = splitapply(@max,data2.chol,G);
MinChol = splitapply(@min,data2.chol,G);

MeanTrestbps = splitapply(@mean,data2.Trestbps,G);
StdTrestbps = splitapply(@std,data2.Trestbps,G);
SkewTrestbps = splitapply(@skewness,data2.Trestbps,G);
MaxTrestbps = splitapply(@max,data2.Trestbps,G);
MinTrestbps = splitapply(@min,data2.Trestbps,G);

Meanthalach = splitapply(@mean,data2.thalach,G);
Stdthalach = splitapply(@std,data2.thalach,G);
Skewthalach = splitapply(@skewness,data2.thalach,G);
Maxthalach = splitapply(@max,data2.thalach,G);
Minthalach = splitapply(@min,data2.thalach,G);

%Add visualization/Excel table for these descriptive statistics

%Graphs
%Age distribution
figure(1)
hold on
h1age = histogram(data2.Age(data.Target==1,:), 20, 'Normalization','probability');
h2age = histogram(data2.Age(data.Target==0,:), 20, 'Normalization','probability');
title('Age distribution')
legend('Disease','No Disease')
hold off

%Resting blood Pressure distribution
figure(2)
hold on
h1duration = histogram(data.Trestbps(data.Target==1,:),30,'Normalization','probability');
h2duration = histogram(data.Trestbps(data.Target==0,:),30,'Normalization','probability');
title('Resting blood Pressure distribution')
legend('Disease','No Disease')
hold off

%% SMOTE


%From 45%-5 5% to 50%-50% duplicating desease observations. The Dataset is
%slightly unbalanced, but it is in general a good practice in health care. 
%Usually the number of observations with a specific disease are only a small
%portion of the population (dataset)

%data split
data_yes = data2(data2.Target==1,:);
data_no = data2(data2.Target==0,:);

%
[yes_rows yes_col]=size(data_yes)

%random process initialised, in this way the 
rng(45)

%duplication of 5% of rows with disease randomly chosen
for i= 1:25
    
    n=randi(yes_rows)
    table(i,:)= data_yes(n,:)
    
end

%Merging of the new rows
data3= [data2;table]

%Rows shuffled, doing so we don't have the duplicate Rows at the end of
%the matrix but their mixed with the other observations.
data_end = data3(randperm(size(data3,1)), :)

%dataset is saved
writetable(data_end,'clean.csv')

