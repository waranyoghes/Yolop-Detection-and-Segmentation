# Yolop-Detection-and-Segmentation
###### This repo contains the code for the detection and segmentation using a single network.
##### Network Details
###### The network developed in this repo is a model based on the YOLOP paper (https://arxiv.org/pdf/2108.11250.pdf).
##### Netwok structure:
![image](https://user-images.githubusercontent.com/73269696/180720374-bb42f26e-3bfc-4116-9299-d0f09cbcb7bd.png)

###### Backbone - CSPDarknet53 
###### Neck - Feature pyramid Module(https://arxiv.org/abs/1612.03144) and SPP (https://arxiv.org/abs/1406.4729v4). 
###### Detection Block - Path aggregation network (https://arxiv.org/abs/1803.01534). 
###### Segemetation Block - U-net https://arxiv.org/abs/1505.04597. 


