_base_ = [
    '../_base_/models/fcn_hr48.py',
    '../_base_/datasets/ddr_crop.py',  # modify
    # '../_base_/datasets/ddr_crop_aug.py',  # modify
    # '../_base_/datasets/ddr_crop_nocolor.py',  # modify
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
model = dict(
    type='EncoderDecoder_Lesion',
    crop_info_path='../data/DDR/eval_crop.txt',  # modify
    decode_head=dict(
        num_classes=4,
        loss_decode=dict(type='BinaryLoss', loss_weight=1.0, loss_type='dice', smooth=1e-5)
    )
)
