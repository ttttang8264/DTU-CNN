% written by Renjia
% 2021-12-27

% 2D CNN
% test input: radar image: 32(length)*32*52(channel)*samples

clear;
close all;
clc;
tic
testno = 10;         % testing number

filename = 'TE_Radar_mul.xls';

trnum=480;
testnum=800;
weight=32;
height=32;

trainx=zeros(weight,height,1,trnum*21);
testx=zeros(weight,height,1,testnum*21);

for index = 1:21  %21   % index: fault no.
    for i=1:trnum
        radarname1=strcat('D:\radar_graph3\',sprintf('d%02i',index),'_', num2str(i),'.png');
        im=imread(radarname1);
%       imshow(im);
        im=rgb2gray(im);
        im=double(im);
        im=im./255;
        im=imresize(im,[weight,height]);
        trainx(:,:,1,(index-1)*trnum+i)=im(:,:);
    end

    for i=161:160+testnum
        radarname2=strcat('D:\radar_graph3\',sprintf('d%02i',index),'_te_', num2str(i),'.png');
        im=imread(radarname2);
%         imshow(im);
        im=rgb2gray(im);
        im=double(im);
        im=im./255;
        im=imresize(im,[weight,height]);
        testx(:,:,1,(index-1)*testnum+i-160)=im(:,:);
    end
    trainy ((index-1)*trnum+1:(index-1)*trnum+trnum,1) = index;
    testy ((index-1)*testnum+1:(index-1)*testnum+testnum,1)= index;
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

