import torch
import torch.nn.functional as F

from .encoder_decoder import EncoderDecoder
from ..builder import SEGMENTORS

import os
import numpy as np

@SEGMENTORS.register_module()
class EncoderDecoder_Lesion(EncoderDecoder):
    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,

                 use_sigmoid=True,
                 compute_aupr=True,
                 crop_info_path=None,
                 ):
        super(EncoderDecoder_Lesion, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

        self.use_sigmoid = use_sigmoid
        self.compute_aupr = compute_aupr

        self.crop_info_dataset = None
        if crop_info_path is not None:
            assert os.path.exists(crop_info_path), f'crop info file {crop_info_path} not exist'
            self.crop_info_dataset = self.load_crop_info_dataset(crop_info_path)

    @staticmethod
    def load_crop_info_dataset(path):
        dataset = dict()
        with open(path) as f:
            lines = f.readlines()
            for line in lines:
                item = line.strip().split(' ')
                dataset[item[0]] = [int(i) for i in item[1:]]
        return dataset

    def restore_pad_label(self, seg_logit, img_meta):
        """
        Args:
            seg_logit: [num_class, H, W] range:[0.,1.]
        """
        filename = os.path.basename(img_meta[0]['filename']).split('.')[0]
        top, bottom, left, right, ori_h, ori_w = self.crop_info_dataset[filename]
        seg_logit = np.pad(seg_logit, ((0, 0), (top, bottom), (left, right)), mode='constant', constant_values=0)

        _, h, w = seg_logit.shape
        assert ori_h == h and ori_w == w, 'restore error in {}'.format(img_meta[0]['filename'])

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)

        # NEW
        if self.use_sigmoid:
            output = torch.sigmoid(seg_logit)
        else:
            output = F.softmax(seg_logit, dim=1)
        # END NEW

        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3,))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2,))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)

        # NEW
        if self.use_sigmoid:
            if self.compute_aupr:
                seg_logit = seg_logit.squeeze(0).cpu().numpy()
                if self.crop_info_dataset is not None:
                    seg_logit = self.restore_pad_label(seg_logit, img_meta)
                seg_logit = [(seg_logit, self.use_sigmoid, self.compute_aupr)]
            else:
                seg_logit = (seg_logit > 0.5).int()
                seg_logit = seg_logit.squeeze(0).cpu().numpy()
                if self.crop_info_dataset is not None:
                    seg_logit = self.restore_pad_label(seg_logit, img_meta)
            return seg_logit
        # END NEW

        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    # TODO: untested
    def aug_test(self, imgs, img_metas, rescale=True):
        return super(EncoderDecoder_Lesion, self).aug_test(imgs, img_metas, rescale)
