If you fail to install and run this tracker, please email me (<zhangyunhua@mail.dlut.edu.cn>)

# Introduction

This repository includes tensorflow code of **MBMD** (MobileNet-based tracking by detection algorithm) for VOT2018 Long-Term Challenge.  
The corresponding arxiv paper has been drafted on Arxiv.  
[Learning regression and verification networks for long-term visual tracking](https://arxiv.org/abs/1809.04320).  
Yunhua Zhang, Dong Wang, Lijun Wang, Jinqing Qi, Huchuan Lu

# Prerequisites

python 2.7  
ubuntu 14.04  
cuda-8.0  
cudnn-6.0.21  
[Tensorflow-1.3-gpu](https://mirrors.tuna.tsinghua.edu.cn/tensorflow/linux/gpu/tensorflow_gpu-1.3.0rc0-cp27-none-linux_x86_64.whl)  
NVIDIA TITAN X GPU

# Pretrained model

The bounding box regression's architecture is MobileNet, and the verifier's architecture is VGGM.   
The pre-trained model can be downloaded at https://drive.google.com/open?id=1g3aMRi6CWK88FOEYoQjqs61fY6QvGW1Z.  
Then you should copy the two files to the folder of our code.  

# Integrate into VOT-2018

The interface for integrating the tracker into the vot evaluation tool kit is implemented in the module `python_long_MBMD.py`. The script `tracker_MBMD.m` is needed to be copied to vot-tookit.  

# CPU manner
If you want to run this code on CPU, you need to just set os.environ \["CUDA_VISIBLE_DEVICES"\]="" in the begin of `python_long_MBMD.py`
