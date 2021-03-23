_base_ = [
    './fcn_hr48_40k_idrid_bdice.py'
]
runner = dict(type='IterBasedRunner', max_iters=10)
evaluation = dict(interval=10, metric='mIoU')
log_config = dict(interval=10)
