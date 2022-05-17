% revised by Renjia
% 2021-11-24


% 1D CNN
% input size: 52(length)*1(width)*1(channel)*(samples)

clear;
close all;
clc;
tic
testno = 10;            % number of testing
a=load('d00.dat');     % load the normal data
a=a';                  % 500*52


for caseno = 1: 4   % caseno=1: 1-52 original；2：random；3：pearson；4：MIC

for index1 = 1:21   % index: fault no
      
    name_fault_train = strcat(sprintf('d%02i',index1),'.dat');      
    name_fault_test = strcat(sprintf('d%02i',index1),'_te.dat');

    b=load(name_fault_train); % load the fault training data 480*52
    c=load(name_fault_test);  % load the fault test data 960*52
    trainx=[a;b];     %980*52
    testx=c;
    trainy=zeros(size(trainx,1),1); % set fault:1，normal:0
    trainy(size(a,1)+1:size(trainx,1))=1;
    testy=zeros(size(testx,1),1);
    testy(161:size(testx,1))=1;     % the last 800 data is faulty data

%     trainx=(trainx-min(trainx))./(max(trainx)-min(trainx));% normalization
%     testx=(testx-min(testx))./(max(testx)-min(testx));
    trainx=(trainx-min(a))./(max(a)-min(a));% normalization
    testx=(testx-min(a))./(max(a)-min(a));
    switch caseno
        case 2
           % rank randomly
           r=randperm(size(trainx,2));   % generate random sequence
           trainx=trainx(:,r);           % reorder of trainx
           testx=testx(:,r);             % reorder of testx
        case 3
           % rank according to pearson coefficients
            X=trainx(1:500,:);
            A=corrcoef(X);
            [A,order]=sort(abs(A(1,:)),'descend');
            trainx=trainx(:,order);
            testx=testx(:,order); 
        case 4
            % rank according to MIC
            load mic;        
            [mic1,order]=sort(abs(mic(1,:)),'descend');
            trainx=trainx(:,order);
            testx=testx(:,order);
        otherwise
    end
 
    % convert the data into the CNN input
    trainx=reshape(trainx',size(trainx,2),1,1,size(trainx,1));
    testx=reshape(testx',size(testx,2),1,1,size(testx,1));
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
    xlslocation2 = strcat('B',num2str(index1+22),':K',num2str(index1+22));
    writematrix(acc,'2TE_f_1_1.xls','Sheet',caseno,'Range',xlslocation);
    writematrix(f1,'2TE_f_1_1.xls','Sheet',caseno,'Range',xlslocation2);
end
end
toc

