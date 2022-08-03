# Using Deep Convolution Neural Network and ELM realizing Image recognition on big dataset 
## Introduction
1.Used pretrained model to extract deep feature, deep feature extracted from the layer before the last output fully-connect layer.

2.Combined 3 deep feature maps together as one deep feature map of from 3 pretrained DCNN models. (P.S.The feature map extracted from those 3 models has same dimension, so we can combine them directly.)

3.ELM was being choosed as classifier to train the deep feature.

- 3 Pretained DCNN models: resnet101,xception,inception-v3
- Database choosed: Caltech101,Caltech256
- Environment: Matlab

## Result
After 3 runs

- Caltech101 Average TestingAccuracy: (94.26%+94.28%+94.41%)/3=94.32%
- Caltech256 Average TestingAccuracy: (88.32%+88.5%+88.71%)/3=88.51%
