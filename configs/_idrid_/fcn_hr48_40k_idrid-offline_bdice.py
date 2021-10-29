_base_ = [
    '../_base_/models/fcn_hr48.py',
    '../_base_/datasets/idrid_offline.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
model = dict(
    type='EncoderDecoder_Lesion',
    decode_head=dict(
        num_classes=4,
        loss_decode=dict(type='BinaryLoss', loss_weight=1.0, loss_type='dice', smooth=1e-5)
    )
)
