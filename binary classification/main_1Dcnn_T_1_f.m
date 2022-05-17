% wirtten by Renjia
% 2021-12-24

% 1D CNN
% input size: T(length)*1*F（feature-channel)*samples
% stride = 1;

clear;
close all;
clc;
tic
testno = 10;           % number of testing
a=load('d00.dat');     % load the normal data
a=a';                  % size: 500*52

T=40; % 20, 30,35,40
filename = strcat('2TE_T',num2str(T),'_1_f.xls');
for index1 = 1:21   % index: fault no
      
    name_fault_train = strcat(sprintf('d%02i',index1),'.dat');      
    name_fault_test = strcat(sprintf('d%02i',index1),'_te.dat');
    b=load(name_fault_train); % load the fault training data 480*52
    c=load(name_fault_test);  % load the fault test data 960*52
    a_b = [a;b];
%     a1 = (a-min(a_b))./(max(a_b)-min(a_b));% normalization
%     b = (b-min(a_b))./(max(a_b)-min(a_b));
%     c = (c-min(c))./(max(c)-min(c));
    a1 = (a-min(a))./(max(a)-min(a));% normalization
    b = (b-min(a))./(max(a)-min(a));
    c = (c-min(a))./(max(a)-min(a));
    trainx= zeros(T,1,52,size(a1,1)-T+size(b,1)-T+2);
    for i = 1:size(a1,1)-T+1
        trainx(:,:,:,i)=a1(i:i+T-1,:);
    end

    for j = 1:size(b,1)-T+1
        trainx(:,:,:,j+i)=b(j:j+T-1,:);
    end

    trainy = [zeros(i,1);ones(j,1)];
    testx= zeros(T,1,52,size(c,1)-T-T+2);
    for k = 1:(160-T+1)
        testx(:,:,:,k)=c(k:k+T-1,:);  
    end
    testy = zeros(k,1);
    for k2 = 161:size(c,1)-T+1
        testx(:,:,:,k+k2-160)=c(k2:k2+T-1,:);
    end
    testy = [testy; ones(k2-160,1)];
    trainy=categorical(trainy);%turn label into catogory
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

        fullyConnectedLayer(2)% output layer
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

    acc=zeros(1,testno);
    f1 = zeros(1,testno);
    for i=1:testno
        % train the net
        net = trainNetwork(trainx,trainy,layers,options);% train
        % test
        py=net.classify(testx);          % predict
        acc(i)=sum(py==testy)/length(py); % accuracy
        [cm,~] = confusionmat(testy,py);
        precision = diag(cm)./sum(cm,2);
        recall = diag(cm)./sum(cm,1)';
        precision = mean(precision);
        recall= mean(recall);
        f1(i) = 2*precision*recall/(precision+recall);% macro F1score
    end
    xlslocation = strcat('B',num2str(index1+1),':K',num2str(index1+1));
    xlslocation2 = strcat('B',num2str(index1+23),':K',num2str(index1+23));
    writematrix(acc,filename,'Sheet',1,'Range',xlslocation);
    writematrix(f1,filename,'Sheet',1,'Range',xlslocation2);
end
toc

