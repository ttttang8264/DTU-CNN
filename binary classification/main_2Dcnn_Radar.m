% written by Renjia
% 2021-12-30

% 2D CNN
% test input: radar image: 32(length)*32*1(channel)*samples

clear;
close all;
clc;
tic
testno = 10;         % testing number

filename = '2TE_Radar.xls';

trnum0=500;
trnum1=480;
testnum0=160;
testnum1=800;
weight=32;
height=32;

trainx=zeros(weight,height,1,trnum0+trnum1);
trainy=ones(trnum0+trnum1,1);
trainy(1:trnum0)=0;
testx=zeros(weight,height,1,testnum0+testnum1);
testy=ones(testnum0+testnum1,1);
testy(1:testnum0)=0;

for i=1:trnum0
        radarname1=strcat('D:\radar_graph3\','d00_', num2str(i),'.png');
        im=imread(radarname1);
%       imshow(im);
        im=rgb2gray(im);
        im=double(im);
        im=im./255;
        im=imresize(im,[weight,height]);
        trainx(:,:,1,i)=im(:,:);
end

for index1 = 1:21  %21   % index: fault no.

    for i=1:trnum1
        radarname1=strcat('D:\radar_graph3\',sprintf('d%02i',index1),'_', num2str(i),'.png');
        im=imread(radarname1);
%       imshow(im);
        im=rgb2gray(im);
        im=double(im);
        im=im./255;
        im=imresize(im,[weight,height]);
        trainx(:,:,1,trnum0+i)=im(:,:);
    end

    for i=1:testnum0+testnum1
        radarname2=strcat('D:\radar_graph3\',sprintf('d%02i',index1),'_te_', num2str(i),'.png');
        im=imread(radarname2);
%         imshow(im);
        im=rgb2gray(im);
        im=double(im);
        im=im./255;
        im=imresize(im,[weight,height]);
        testx(:,:,1,i)=im(:,:);
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
%     writematrix(acc,filename,'Sheet',1,'Range',xlslocation);
%     writematrix(f1,filename,'Sheet',1,'Range',xlslocation2);
     xlswrite(filename,acc,1,xlslocation);
     xlswrite(filename,f1,1,xlslocation2);
end
toc

