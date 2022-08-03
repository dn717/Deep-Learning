# Using Deep Convolution Neural Network and ELM realizing Image recognition on big dataset 
## Introduction
Used pretrained model to extract deep feature, deep feature extracted from the layer before the last output fully-connect layer.
Combined 3 deep feature maps together as one deep feature map of from 3 pretrained DCNN models. (P.S.The feature map extracted from those 3 models has same dimension, so we can combine them directly.)

- 3 Pretained DCNN models: resnet101,xception,inception-v3
