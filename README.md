# SpineParseNet: Spine Parsing for Volumetric MR Image by a 	Two-Stage Segmentation Framework with Semantic Image Representation

This repository is the official implementation of the paper:

Shumao Pang, Chunlan Pang, Lei Zhao, Yangfan Chen, Zhihai Su, Yujia Zhou, Meiyan Huang, Wei Yang, Hai Lu, and Qianjin Feng, "[SpineParseNet: Spine Parsing for Volumetric MR Image by a Two-Stage Segmentation Framework with Semantic Image Representation](https://ieeexplore.ieee.org/document/9201093), " IEEE Transactions on Medical Imaging, 2021, doi: 10.1109/TMI.2020.3025087.

Website: https://www.researchgate.net/profile/Shumao_Pang2

Some codes were borrowed from 
+ https://github.com/wolny/pytorch-3dunet
+ https://github.com/Gaoyiminggithub/Graphonomy

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
+ The dataset can be downloaded at: https://www.spinesegmentation-challenge.com
+ The data structure should look like:

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

# Citation
@article{pang2020spineparsenet,
  title={SpineParseNet: Spine Parsing for Volumetric MR Image by a Two-Stage Segmentation Framework With Semantic Image Representation},
  author={Pang, Shumao and Pang, Chunlan and Zhao, Lei and Chen, Yangfan and Su, Zhihai and Zhou, Yujia and Huang, Meiyan and Yang, Wei and Lu, Hai and Feng, Qianjin},
  journal={IEEE Transactions on Medical Imaging},
  volume={40},
  number={1},
  pages={262--273},
  year={2021},
  publisher={IEEE}
}
