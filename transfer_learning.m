Folder=fullfile('F:\DATA\Caltech256');
rootFolder=fullfile(Folder);
imds=imageDatastore(Folder,'IncludeSubfolders',true,"LabelSource","foldernames");

%%
[trainData,testData]=splitEachLabel(imds,30,"randomized");
%%
%trainData=augmentedImageDatastore([224,224],trainData);
%testData=augmentedImageDatastore([224,224],testData);

trainData.ReadFcn=@customReadDatastoreImage;
testData.ReadFcn=@customReadDatastoreImage;

%%
net=vgg16;
layersTransfer=net.Layers(1:end-3);
numClasses=257;

Layers=[
    layersTransfer
    fullyConnectedLayer(numClasses,'Name','fc3')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')
    ];
%%
for i=1:3
    miniBatchsize=15;
    validationFrequency=floor(numel(trainData.Labels)/miniBatchsize);
    options=trainingOptions('sgdm','LearnRateSchedule',"piecewise", ...
        "MiniBatchSize",miniBatchsize, ...
        'LearnRateDropFactor',0.1, ...
        'LearnRateDropPeriod',10, ...
        'MaxEpochs',10, ...
        'InitialLearnRate',0.0001);
    
    gpuDevice(1)
    convnet=trainNetwork(trainData,Layers,options);
    
    Ypred=classify(convnet,trainData,'MiniBatchSize',15);
    Ytest=trainData.Labels;
    training_accuracy(i)=sum(Ypred==Ytest)/numel(Ytest)
    
    Ypred=classify(convnet,testData);
    Ytest=testDat
a.Labels;
    testing_accuracy(i)=sum(Ypred==Ytest)/numel(Ytest)
end
%%
function data = customReadDatastoreImage(filename)
% code from default function: 
onState = warning('off', 'backtrace'); 
c = onCleanup(@() warning(onState)); 
data = imread(filename); % added lines: 
data = data(:,:,min(1:3, end)); 
data = imresize(data,[224 224]);
end