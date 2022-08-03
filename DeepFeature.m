Folder=fullfile('F:\DATA\Caltech256');
imds=imageDatastore(Folder,'IncludeSubfolders',true,"LabelSource","foldernames");
%%
[trainData,testData]=splitEachLabel(imds,30,"randomized");
trainData.ReadFcn=@customReadDatastoreImage;
testData.ReadFcn=@customReadDatastoreImage;
%%
net=vgg16;
%%
gpuDevice(1)
trainingFeatures=activations(net,trainData,'drop7','MiniBatchSize',30);
trainingFeatures=reshape(trainingFeatures,[size(trainingFeatures,3),size(trainingFeatures,4)])';

testingFeatures=activations(net,testData,'drop7','MiniBatchSize',30);
testingFeatures=reshape(testingFeatures,[size(testingFeatures,3),size(testingFeatures,4)])';
%%
train_label=grp2idx(trainData.Labels);
test_label=grp2idx(testData.Labels);
%%
Training=[train_label trainingFeatures];
Testing=[test_label testingFeatures];
%%
function data = customReadDatastoreImage(filename)
% code from default function: 
onState = warning('off', 'backtrace'); 
c = onCleanup(@() warning(onState)); 
data = imread(filename); % added lines: 
data = data(:,:,min(1:3, end)); 
data = imresize(data,[224 224]);
end