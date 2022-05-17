% written by Renjia
% Zhejiang Sci-Tech University, Hangzhou, China
% 2021-12-31


% 1D CNN
% input size: 52(Features)*1(width)*1(channel)*(samples)


clear;
close all;
clc;
tic

testno = 10;         % number of testing   
filename = 'pearson_TE_F_1_1_mul.xls';

load mic;
[~,order]=sort(abs(mic(1,:)),'descend');

[fault_train, fault_test]=load_normalize( );
Len1 = 480;
Len2 = 800;
trainx= zeros(52,1,1,Len1*21);
testx= zeros(52,1,1, Len2*21);

for index = 1:21   % index: fault no.
    a1 = fault_train{index};
    b   = fault_test{index};
    a1 = a1(:,order);
    b   = b (:,order);
%     X=load('d00.dat');
%     X=X';
%     A=corrcoef(X);
%     [A,order]=sort(abs(A(1,:)),'descend');
%     a1=a1(:,order);
%     b=b(:,order);
    trainx(:,1,1,(index-1)*Len1+1:(index-1)*Len1+Len1)=a1';
    testx(:,1,1,(index-1)*Len2+1:(index-1)*Len2+Len2)=b';   
    trainy ((index-1)*Len1+1:(index-1)*Len1+Len1,1) = index;
    testy ((index-1)*Len2+1:(index-1)*Len2+Len2,1)= index;
end


trainy=categorical(trainy);%turn label into category
testy=categorical(testy);

layers = [
imageInputLayer([size(trainx,1) size(trainx,2) size(trainx,3)])%CNN input

convolution2dLayer([3 1],16,'Padding','same')% convolutional layer
batchNormalizationLayer
reluLayer

maxPooling2dLayer([2 1],'Stride',2)%pooling layer

convolution2dLayer([3 1],16,'Padding','same')% convolutional layer 
batchNormalizationLayer
reluLayer

maxPooling2dLayer([2 1],'Stride',2)%pooling layer

fullyConnectedLayer(64) %full connected layer
batchNormalizationLayer
reluLayer

fullyConnectedLayer(21)% output layer
softmaxLayer
classificationLayer];

options = trainingOptions('sgdm', ...% training alogrithm:sgdm
    'MiniBatchSize',100, ...%batchsize
    'MaxEpochs', 16 , ...   % maximal epochs
    'InitialLearnRate',0.7, ...
     'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20, ...
    'Shuffle','every-epoch', ...
     'ValidationPatience', Inf,...
     'ExecutionEnvironment','cpu',...
     'GradientThreshold',10,...
    'Verbose',false);


for i=1:testno
    % train the net
    net = trainNetwork(trainx,trainy,layers,options);% train
    % test
    py=net.classify(testx);          % predict
    
    [cm,~] = confusionmat(testy,py);
    precision = diag(cm)./sum(cm,2);
    recall = diag(cm)./sum(cm,1)';
    f1 = 2*precision.*recall./(precision+recall+0.00001);% macro F1score
    
%     write the data into excel file
    xlslocation = strcat('B',num2str(i+1),':V',num2str(i+1));
    xlslocation1 = strcat('B',num2str(i+12),':V',num2str(i+12));
    xlslocation2 = strcat('B',num2str(i+24),':V',num2str(i+24));
%     writematrix(precision',filename,'Sheet',1,'Range',xlslocation);
%     writematrix(recall',filename,'Sheet',1,'Range',xlslocation1);
%     writematrix(f1',filename,'Sheet',1,'Range',xlslocation2);
     xlswrite(filename,precision',1,xlslocation);
     xlswrite(filename,recall',1,xlslocation1);
     xlswrite(filename,f1',1,xlslocation2);
end

toc  

