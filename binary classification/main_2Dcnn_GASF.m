% written by Renjia
% 2021-12-24

% 2D CNN
% test input: GASF image: T*T*F(channel)*samples
% stride = 1;
% T: width of sliding time window；

clear;
close all;
clc;
tic
testno = 10;         % testing number
T = 40;              % width of sliding time window:20,30,35,40
filename = strcat('2TE_GASF_T',num2str(T),'new.xls');

a=load('d00.dat');     %load the training data
a=a';                  %500*52
mina = min(a);
maxa = max(a);

for index1 = 1:21   % index: fault no.
      
    name_fault_train = strcat(sprintf('d%02i',index1),'.dat');      
    name_fault_test = strcat(sprintf('d%02i',index1),'_te.dat');
    b=load(name_fault_train);    % load the training data of fault 480*52
    c=load(name_fault_test);     % load the test data of fault 960*52
    a1 = (a-mina)./(maxa-mina);%normalization
    b = (b-mina)./(maxa-mina);
    c = (c-mina)./(maxa-mina);
    a1 = 1./(1+exp(-a1));
    b = 1./(1+exp(-b));
    c = 1./(1+exp(-c));
    trainx= zeros(T,T,52,size(a1,1)-T+size(b,1)-T+2);
    for i = 1:size(a1,1)-T+1
        for k = 1 : size(a1,2) 
            t = a1(i:i+T-1,k);
            t1 = sqrt(1-t'.^2);
            trainx(:,:,k,i) = t'*t-t1'*t1;
        end
    end

    for j = 1:size(b,1)-T+1
        for k = 1:size(b,2)
            t = b(j:j+T-1,k);
            t1 = sqrt(1-t'.^2);
            trainx(:,:,k,i+j)= t'*t-t1'*t1;
        end
    end

    trainy = [zeros(i,1);ones(j,1)];
    testx= zeros(T,T,52,size(c,1)-T-T+2);
    for k1 = 1:(160-T+1)
        for k = 1:size(c,2)
            t = c(k1:k1+T-1,k);
            t1 = sqrt(1-t'.^2);
            testx(:,:,k,k1)= t'*t-t1'*t1;
        end
    end
    testy = zeros(k1,1);
    for k2 = 161:size(c,1)-T+1
        for k = 1:size(c,2)
            t = c(k2:k2+T-1,k);
            t1 = sqrt(1-t'.^2);
            testx(:,:,k,k1+k2-160)= t'*t-t1'*t1;
        end
    end
    testy = [testy; ones(k2-160,1)];

    trainy=categorical(trainy);%turn label into catogory
    testy=categorical(testy);

    layers = [
        imageInputLayer([size(trainx,1) size(trainx,2) size(trainx,3)])%CNN input

        convolution2dLayer([3 3],16,'Padding','same')% convolutional layer
        batchNormalizationLayer
        reluLayer

        maxPooling2dLayer([2 2],'Stride',2)%pooling layer

        convolution2dLayer([3 3],16,'Padding','same')% convolutional layer 
        batchNormalizationLayer
        reluLayer

        maxPooling2dLayer([2 2],'Stride',2)%pooling layer

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
        f1(i) = 2*precision*recall/(precision+recall+0.000001);% macro F1score
    end
    xlslocation = strcat('B',num2str(index1+1),':K',num2str(index1+1));
    xlslocation2 = strcat('B',num2str(index1+23),':K',num2str(index1+23));
    writematrix(acc,filename,'Sheet',1,'Range',xlslocation);
    writematrix(f1,filename,'Sheet',1,'Range',xlslocation2);
end
toc

