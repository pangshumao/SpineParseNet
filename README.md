# SpineParseNet: Spine Parsing for Volumetric MR Image by a 	Two-Stage Segmentation Framework with Semantic Image Representation

This repository contains the code for the paper:

Shumao Pang, Chunlan Pang, Lei Zhao, Yangfan Chen, Zhihai Su, Yujia Zhou, Meiyan Huang, Wei Yang, Hai Lu, and Qianjin Feng, "SpineParseNet: Spine Parsing for Volumetric MR Image by a Two-Stage Segmentation Framework with Semantic Image Representation, " IEEE Transactions on Medical Imaging, 2020.

![image](https://github.com/pangshumao/SpineParseNet/blob/master/Figures/Spine_parsing.gif)

# Environment and installation
+ Pytorch = 1.5.1
+ torchvision
+ scipy
+ tensorboardX
+ numpy
+ opencv-python
+ matplotlib
+ networkx

# Getting Started
### Data Preparation
+ The file structure should look like:

|-- data
>|-- coarse
>>|-- in
>>>|-- h5py

>>>|-- nii
>>>>|-- original_mr
>>>>>|-- Case1.nii.gz

>>>>>|-- Case2.nii.gz

>>>>>......

>>>>>|-- Case215.nii.gz
>>>>
>>>>|-- mask
>>>>>|-- mask_case1.nii.gz

>>>>>|-- mask_case2.nii.gz

>>>>>......

>>>>>|-- mask_case215.nii.gz
>
>|-- fine
>>|-- in
>>>|-- h5py

>>>|-- nii
>>>>|-- original_mr
>>>>>|-- Case1.nii.gz

>>>>>|-- Case2.nii.gz

>>>>>......

>>>>>|-- Case215.nii.gz
>>>>
>>>>|-- mask
>>>>>|-- mask_case1.nii.gz

>>>>>|-- mask_case2.nii.gz

>>>>>......

>>>>>|-- mask_case215.nii.gz

Note that the files in data/coarse/in/nii are the same with those in data/fine/in/nii.

### Run the main.sh script for training and test:
nohup ./main.sh > main.out &

