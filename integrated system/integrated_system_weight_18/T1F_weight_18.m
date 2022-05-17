function py = T1F_weight_18(fault_train, fault_test,order)
T=40;
Len1 = 480-T+1;
Len2 = 800-T+1;
trainx = zeros(T,1,52, Len1*21);
testx  = zeros(T,1,52, Len2*21);


for index = 1:21   % index: fault no
    a1 = fault_train{index};
    b   = fault_test{index};
    a1 = a1(:,order);
    b   = b (:,order);
    for i = 1: Len1
        trainx(:,:,:,(index-1)*Len1+i)=a1(i:i+T-1,:);
    end

    for k = 1:Len2
        testx(:,:,:,(index-1)*Len2+k)=b(k:k+T-1,:);  
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
% testy=categorical(testy);

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
    py=activations(net,testx,'classoutput');       % predict
end
