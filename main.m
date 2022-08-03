Folder=fullfile('G:\Final_project\Caltech101');
imds=imageDatastore(Folder,'IncludeSubfolders',true,"LabelSource","foldernames");
%%
[trainData,testData]=splitEachLabel(imds,30,"randomized");
%%
%change the image size 224*224 for resnet and densenet,299*299 for xception and inception
trainData.ReadFcn=@customReadDatastoreImage; 
testData.ReadFcn=@customReadDatastoreImage;
%%
%net=resnet101;%224*224*3
%net=xception; %299*299*3
net=inceptionv3; %299*299*3
%net = inceptionresnetv2; %299*299*3
%%
%%% Extract Deep Feature from DCNN pretrained model %%%
%gpuDevice(1)

%xception,inception-v3
trainingFeatures=activations(net,trainData,'avg_pool','MiniBatchSize',100);

%resnet101
%trainingFeatures=activations(net,trainData,'pool5','MiniBatchSize',200);
trainingFeatures=reshape(trainingFeatures,[size(trainingFeatures,3),size(trainingFeatures,4)])';
%%
testingFeatures=activations(net,testData,'avg_pool','MiniBatchSize',100);
%testingFeatures=activations(net,testData,'pool5','MiniBatchSize',200);

testingFeatures=reshape(testingFeatures,[size(testingFeatures,3),size(testingFeatures,4)])';
%%
%%%Get Label%%%
train_label=grp2idx(trainData.Labels);
test_label=grp2idx(testData.Labels);
%%
%%combine deep-trainfea and deep-testfea
resnet_totalfea=[resnet_trainfea;resnet_testfea];
xception_totalfea=[xception_trainfea;xception_testfea];
inception_totalfea=[inception_trainfea;inception_testfea];
%%
%%%Combine extracted feature maps%%%

% Way1: get the max value between two feature maps
%temp=max(resnet_totalfea,xception_totalfea);
%total_feature=max(temp,inception_totalfea);
%%
% Way2:concatenate three feature maps
total_feature=[resnet_totalfea xception_totalfea inception_totalfea];
%%
train_feature=total_feature(1:size(train_label),:);
test_feature=total_feature(size(train_label)+1:end,:);

%%

Training=[train_label train_feature];
Testing=[test_label test_feature];
%%
Training=cast(Training,'double');
Testing=cast(Testing,'double');
%%
ELM(Training,Testing,1,7000,'sig',2^1e-10)
%%
function data = customReadDatastoreImage(filename)
% code from default function: 
onState = warning('off', 'backtrace'); 
c = onCleanup(@() warning(onState)); 
data = imread(filename); % added lines: 
data = data(:,:,min(1:3, end)); 
data = imresize(data,[299 299]);
end