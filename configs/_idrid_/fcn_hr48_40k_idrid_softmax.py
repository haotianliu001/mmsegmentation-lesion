_base_ = [
    '../_base_/models/fcn_hr48.py',
    '../_base_/datasets/idrid.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
model = dict(
    decode_head=dict(num_classes=5)
)