_base_ = [
    '../_base_/models/fcn_hr48.py',
    '../_base_/datasets/ddr.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_60k.py'
]
model = dict(
    type='EncoderDecoder_Lesion',
    crop_info_path='../data/DDR/eval_crop.txt',
    decode_head=dict(
        num_classes=4,
        loss_decode=dict(type='BinaryLoss', loss_weight=1.0, loss_type='dice', smooth=1e-5)
    )
)
