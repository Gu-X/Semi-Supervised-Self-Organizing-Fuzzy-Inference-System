clear all
clc
close all

load exampledata.mat
granlevel=10;   % level of granularity
chunksize=1000; % chunk size
datatra=DTra1;  % Labelled training data
labeltra=LTra1; % Labels of the labelled training data
datates=DTes1;  % Unlabelled training data
labeltes=LTes1; % Labels of the unlabelled training data
[EstimatedLabels1,timme]=S3OFIS(datatra,labeltra,datates,granlevel,chunksize); % Run S3OFIS+
CM1=confusionmat(labeltes,EstimatedLabels1); % Confusion matrix
Acc1=sum(sum(confusionmat(labeltes,EstimatedLabels1).*(eye(length(unique(labeltes))))))/length(labeltes); % Classification accuracy on unlabelled training data