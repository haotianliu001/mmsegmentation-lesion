_base_ = [
    '../_base_/models/ocrnet_r50-d8.py',
    '../_base_/datasets/idrid.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='CascadeEncoderDecoder_Lesion',
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=1024,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=4,  # NEW
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(type='BinaryLoss', loss_weight=1.0, loss_type='dice', smooth=1e-5)),  # NEW
        dict(
            type='OCRHead',
            in_channels=2048,
            in_index=3,
            channels=512,
            ocr_channels=256,
            dropout_ratio=0.1,
            num_classes=4,  # NEW
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(type='BinaryLoss', loss_weight=1.0, loss_type='dice', smooth=1e-5))  # NEW
    ],
)
