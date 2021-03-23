# 以PSPNet为例进行说明
# 以下内容由 configs/_base_/models/pspnet_r50-d8.py 修改得到
_base_ = [
    '../_base_/models/pspnet_r50-d8.py',  # 沿用对应的backbone配置文件
    '../_base_/datasets/idrid.py',  # 改变数据集文件，对应idrid.py和ddr.py
    '../_base_/default_runtime.py',  # runtime配置，可以看一下内容
    '../_base_/schedules/schedule_40k.py'  # 关于学习率，迭代次数等的设置
]
model = dict(
    type='EncoderDecoder_Lesion',  # 替换原始的EncoderDecoder类，可以针对眼底数据集训练和预测，计算新的指标等，分类使用sigmoid激活
    # backbone=dict(  # 这里保存不变，有时候会需要修改backbone的配置
    #     type='ResNetV1c',
    #     depth=50,
    # ),
    decode_head=dict(  # 对应最终输出特征图
        num_classes=4,  # 分类使用sigmoid激活，类别数为4，不包含背景类
        loss_decode=dict(type='BinaryLoss', loss_weight=1.0, loss_type='dice', smooth=1e-5)  # 此处是binary dice loss
    ),
    auxiliary_head=dict(  # 对应辅助训练分支
        num_classes=4,
        loss_decode=dict(type='BinaryLoss', loss_weight=0.4, loss_type='dice', smooth=1e-5)  # 此处是binary dice loss
    )
)
