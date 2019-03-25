function [final_features final_mark] = SMOTE(original_features, original_mark)
ind = find(original_mark == 0);
% P = candidate points
P = original_features(ind,:);
T = P';
% X = Complete Feature Vector
X = T;
% Finding the 5 positive nearest neighbours of all the negative blobs
I = nearestneighbour(T, X, 'NumberOfNeighbours', 4);
I = I';
[r c] = size(I);
S = [];
th=0.3;
for i=1:r
    for j=2:c
        index = I(i,j);
        new_P=(1-th).*P(i,:) + th.*P(index,:);
        S = [S;new_P];
    end
end
original_features = [original_features;S];
[r c] = size(S);
mark = zeros(r,1);
original_mark = [original_mark;mark];
train_incl = ones(length(original_mark), 1);
I = nearestneighbour(original_features', original_features', 'NumberOfNeighbours', 4);
I = I';
for j = 1:length(original_mark)
    len = length(find(original_mark(I(j, 2:4)) ~= original_mark(j,1)));
    if(len >= 2)
        if(original_mark(j,1) == 0)
         train_incl(original_mark(I(j, 2:4)) ~= original_mark(j,1),1) = 0;
        else
         train_incl(j,1) = 1;   
        end    
    end
end
final_features1 = original_features(train_incl == 1, :);
final_mark1 = original_mark(train_incl == 1, :);

%% second step
ind2 = find(original_mark == 1);
% P = candidate points
P2 = original_features(ind,:);
T2 = P2';
% X = Complete Feature Vector
X2 = T2;
% Finding the 5 positive nearest neighbours of all the positive blobs
I2 = nearestneighbour(T2, X2, 'NumberOfNeighbours', 4);
I2 = I2';
[r2 c2] = size(I2);
S2 = [];
th=0.3;
for i=1:r2
    for j=2:c2
        index2 = I2(i,j);
        new_P2=(1-th).*P2(i,:) + th.*P2(index,:);
        S2 = [S2;new_P2];
    end
end
original_features2 = [S2];
[r2 c2] = size(S2);
mark = ones(r2,1);
original_mark2 = [mark];
train_incl2 = ones(length(original_mark2), 1);
I2 = nearestneighbour(original_features2', original_features2', 'NumberOfNeighbours', 4);
I2 = I2';
for j = 1:length(original_mark2)
    len = length(find(original_mark2(I2(j, 2:4)) ~= original_mark2(j,1)));
    if(len >= 2)
        if(original_mark2(j,1) == 1)
         train_incl2(original_mark2(I2(j, 2:4)) ~= original_mark2(j,1),1) = 0;
        else
         train_incl2(j,1) = 0;   
        end    
    end
end
final_features2 = original_features2(train_incl2 == 1, :);
final_mark2 = original_mark2(train_incl2 == 1, :);

gamma=0.5
index= 328+randperm((size(final_features1,1)-328),100)
index2=randperm(size(final_features2,1),100)
final_features = [final_features1(1:328,:);final_features1(index,:);final_features2(index2,:)]
final_mark = [final_mark1(1:328,:);final_mark1(index,:);final_mark2(index2,:)]

end