% written by Renjia
% Zhejiang Sci-Tech University, Hangzhou, China
% 2021-12-31

% Fast Fourier Transform+2D CNN
% input size: T(length)*52*1(channel)*samples
% stride = 1;


clear;
close all;
clc;
tic

testno = 1;            
T=40;
filename = strcat('TE_FFT_T',num2str(T),'_mul.xls');

load mic;
[~,order]=sort(abs(mic(1,:)),'descend');

[fault_train, fault_test]=load_normalize( );
Len1 = 480-T+1;
Len2 = 800-T+1;
trainx= zeros(T,52,1,Len1*21);
testx= zeros(T,52,1, Len2*21);

for index = 1:21   % index: fault no.
    a1 = fault_train{index};
    b   = fault_test{index};
    a1 = a1(:,order);
    b   = b (:,order);
    for i = 1: Len1
        temp=a1(i:i+T-1,:);
        trainx(:,:,:,(index-1)*Len1+i) = abs(fft(temp));
    end

    for k1 = 1:Len2
            temp = b(k1:k1+T-1,:);
            testx(:,:,:,(index-1)*Len2+k1)= abs(fft(temp));
    end
    
    trainy ((index-1)*Len1+1:(index-1)*Len1+Len1,1) = index;
    testy ((index-1)*Len2+1:(index-1)*Len2+Len2,1)= index;
end

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
    % cm = confusionchart(testy,py);
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
