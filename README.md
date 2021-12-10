### 摘要

- 数据集配置文件:
  [idrid.py](configs/_base_/datasets/idrid.py)
  ,
  [ddr.py](configs/_base_/datasets/ddr.py)
- 模型配置文件示例:
  [config_sample.py](configs/_idrid_/config_sample.py)
- 一个简单的调试文件：
  [debug.py](configs/_idrid_/debug.py)
  
### 常用命令
#### 训练
```sh
## CUDA_VISIBLE_DEVICES表示使用哪些序号的显卡
## PORT可随意设置，不冲突就行，为程序使用的端口（传数据、syncbn等）
## 最后的数字表示启用几个线程，用n张卡就等于n

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=23456 ./tools/dist_train.sh ./configs/_idrid_/fcn_hr48_40k_idrid_bdice.py 4
# or
CUDA_VISIBLE_DEVICES=2,3 PORT=23456 ./tools/dist_train.sh ./configs/_idrid_/debug.py 2

# 从checkpoint处继续训练
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=23456 ./tools/dist_train.sh ./configs/_idrid_/fcn_hr48_40k_idrid_bdice.py 4 --resume-from ./work_dirs/xxx/xxx.pth
```

#### 测试
```sh
# 以下两种结果应该是一样的
# 多卡
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=23456 ./tools/dist_test.sh ./configs/xxx.py ./work_dirs/xxx/xxx.pth 4 --eval-mIoU
# 单卡
CUDA_VISIBLE_DEVICES=2 python tools/test.py ./configs/xxx.py ./work_dirs/xxx/xxx.pth --eval mIoU
```


### 环境配置

基本环境：

- python=3.7
- torch=1.6.0
- cuda=10.2
- mmcv=1.2.0

```sh
# 创建环境 open-mmlab
conda create -n open-mmlab python=3.7 -y
# 激活环境
conda activate open-mmlab


# 之后的命令一定要在这个环境下
conda install pytorch=1.6.0 torchvision cudatoolkit=10.2 -c pytorch -y
pip install mmcv-full==1.2.0 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.6.0/index.html -i https://pypi.douban.com/simple/
pip install opencv-python -i https://pypi.douban.com/simple/
pip install scipy -i https://pypi.douban.com/simple/
pip install tensorboard tensorboardX -i https://pypi.douban.com/simple/
pip install sklearn -i https://pypi.douban.com/simple/
pip install terminaltables -i https://pypi.douban.com/simple/
pip install matplotlib -i https://pypi.douban.com/simple/

# 以下是cuda 10.1版本的环境
conda install pytorch=1.6.0 torchvision cudatoolkit=10.1 -c pytorch -y
pip install mmcv-full==1.2.0 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html -i https://pypi.douban.com/simple/
pip install opencv-python -i https://pypi.douban.com/simple/
pip install scipy -i https://pypi.douban.com/simple/
pip install tensorboard tensorboardX -i https://pypi.douban.com/simple/
pip install sklearn -i https://pypi.douban.com/simple/
pip install terminaltables -i https://pypi.douban.com/simple/
pip install matplotlib -i https://pypi.douban.com/simple/


cd mmsegmentation-lesion
chmod u+x tools/*
pip install -e . -i https://pypi.douban.com/simple/
```

### 修改的文件列表 
(用于对比和迁移到其他版本的mmsegmentation)

#### new:
- [lesion_metrics.py](mmseg/core/evaluation/lesion_metrics.py)
- [lesion_dataset.py](mmseg/datasets/lesion_dataset.py)
- [encoder_decoder_lesion.py](mmseg/models/segmentors/encoder_decoder_lesion.py)
- [cascade_encoder_decoder_lesion.py](mmseg/models/segmentors/cascade_encoder_decoder_lesion.py)
- [binary_loss.py](mmseg/models/losses/binary_loss.py)

#### modify:
- [evaluation/\_\_init__.py](mmseg/core/evaluation/__init__.py)
- [datasets/\_\_init__.py](mmseg/datasets/__init__.py)
- [segmentors/\_\_init__.py](mmseg/models/segmentors/__init__.py)
- [losses/\_\_init__.py](mmseg/models/losses/__init__.py)

### 附录

anaconda安装

```sh
wget -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2020.02-Linux-x86_64.sh
sh Anaconda3-2020.02-Linux-x86_64.sh
# 然后按提示操作

# 添加环境变量
echo 'export PATH=/root/anaconda3/bin:$PATH' >> ~/.zshrc
source ~/.zshrc
conda init zsh

# 安装包示例
pip install opencv-python
# 或者（使用镜像源）:
pip install opencv-python -i https://pypi.douban.com/simple/
```

anacoda镜像配置（加快conda命令的下载速度，北外为例）

```sh
echo \
'channels:
  - defaults
show_channel_urls: true
channel_alias: https://mirrors.bfsu.edu.cn/anaconda
default_channels:
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/main
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/free
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/r
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/pro
  - https://mirrors.bfsu.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.bfsu.edu.cn/anaconda/cloud
  msys2: https://mirrors.bfsu.edu.cn/anaconda/cloud
  bioconda: https://mirrors.bfsu.edu.cn/anaconda/cloud
  menpo: https://mirrors.bfsu.edu.cn/anaconda/cloud
  pytorch: https://mirrors.bfsu.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.bfsu.edu.cn/anaconda/cloud
'> ~/.condarc
```
