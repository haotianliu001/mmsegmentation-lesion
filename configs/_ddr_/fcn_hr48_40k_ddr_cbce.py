_base_ = [
    '../_base_/models/fcn_hr48.py',
    '../_base_/datasets/ddr.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
model = dict(
    type='EncoderDecoder_Lesion',
    decode_head=dict(
        num_classes=4,
        loss_decode=dict(type='BinaryLoss', loss_weight=1.0,
                         loss_type='ce', class_weight=[1, 1, 1, 1], class_weight_norm=True)
    )
)
