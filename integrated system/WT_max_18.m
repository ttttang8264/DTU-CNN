function py = WT_highest_18(fault_train, fault_test,order)

T=40;
wname = 'db4';
Len1 = 480-T+1;
Len2 = 800-T+1;
trainx= zeros(T,52,3,Len1*21);
testx= zeros(T,52,3, Len2*21);
% trainy = zeros(Len1*21,21);
% testy = zeros(Len2*21,21);

% test_y = zeros(21);

for index = 1:21   % index: fault no.
    a1 = fault_train{index};
    b   = fault_test{index};
    a1 = a1(:,order);
    b   = b (:,order);
    for i = 1: Len1
        temp=a1(i:i+T-1,:);
        for j1 = 1: size(temp,2)
            [C,L]=wavedec(temp(:,j1),2,wname);
            A2=wrcoef('a',C,L,wname,2);
            D1=wrcoef('d',C,L,wname,1);
            D2=wrcoef('d',C,L,wname,2);
            trainx(:,j1,1,(index-1)*Len1+i) = A2;    
            trainx(:,j1,2,(index-1)*Len1+i) = D1;
            trainx(:,j1,3,(index-1)*Len1+i) = D2;
        end
    end
    

    for k1 = 1:Len2
            temp = b(k1:k1+T-1,:);
        for j3 = 1: size(temp,2)
            [C,L]=wavedec(temp(:,j3),2,wname);
            A2=wrcoef('a',C,L,wname,2);
            D1=wrcoef('d',C,L,wname,1);
            D2=wrcoef('d',C,L,wname,2);
            testx(:,j3,1,(index-1)*Len2+k1) = A2;    
            testx(:,j3,2,(index-1)*Len2+k1) = D1;
            testx(:,j3,3,(index-1)*Len2+k1) = D2;
        end
    end
    trainy ((index-1)*Len1+1:(index-1)*Len1+Len1,:) = index;  
end

%Remove faults 3, 9, and 15
A = [3,8,13];
for index1 = A
        trainx(:,:,:,(index1-1)*Len1+1:(index1-1)*Len1+Len1) = [];
        trainy((index1-1)*Len1+1:(index1-1)*Len1+Len1,:) = [];
        testx(:,:,:,(index1-1)*Len2+1:(index1-1)*Len2+Len2) = [];  
end

trainy=categorical(trainy);%turn label into catogory

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

    fullyConnectedLayer(18)% output layer
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
% train the net
    net = trainNetwork(trainx,trainy,layers,options);% train
    % test
    py=net.classify(testx);          % predict
end