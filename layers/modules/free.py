# -*- coding: utf-8 -*-
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable
from torch.autograd.function import once_differentiable
from data import coco as cfg
from ..box_utils import match, log_sum_exp, decode, encode, jaccard, point_form

import numpy as np
torch.set_printoptions(threshold=np.inf)  

class Clip(Function):
    @staticmethod
    def forward(ctx, x, a, b):
        return x.clamp(a, b)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        return grad_output, None, None


clip = Clip.apply

def smooth_l1_loss(pred, target, weight, beta):
    val = target - pred
    abs_val = val.abs()
    smooth_mask = abs_val < beta
    return weight * torch.where(smooth_mask, 0.5 / beta * val ** 2, (abs_val - 0.5 * beta)).sum(dim=-1)


def focal_loss(logits, gamma):
    # print("logits",logits.min(),logits.max())
    return torch.sum(
        logits ** gamma * F.binary_cross_entropy(logits, torch.zeros_like(logits), reduction='none')
    )


def positive_bag_loss(logits, *args, **kwargs):
    # bag_prob = Mean-max(logits)
    weight = 1 / clip(1 - logits, 1e-12, None)
    weight /= weight.sum(*args, **kwargs).unsqueeze(dim=-1)
    bag_prob = (weight * logits).sum(*args, **kwargs)
    # positive_bag_loss = -log(bag_prob)
    return F.binary_cross_entropy(bag_prob, torch.ones_like(bag_prob), reduction='none')


class FreeLoss(nn.Module):

    def __init__(self, num_classes, use_gpu=True):
        super(FreeLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes # 20
        self.bbox_threshold = 0.5
        self.variance = cfg['variance'] # [0.1, 0.2]
        # 每个obj的正样本的数量
        self.pre_anchor_topk = 50 
        self.smooth_l1_loss_param = (0.75, 0.11)
        self.focal_loss_alpha = 0.5
        self.focal_loss_gamma = 2.0
        self.positive_bag_loss_func = positive_bag_loss
        self.negative_bag_loss_func = focal_loss

        # self.softmax = nn.Softmax(dim=-1)


    def forward(self, predictions, targets):
        # batch * anchor size * 4
        # batch * anchor size * numclass
        # anchor size * 4
        box_regression, cls_prob, anchors_ = predictions
        # print(anchors_.size())
        cls_prob = torch.sigmoid(cls_prob)

        box_prob = []
        positive_numels = 0
        positive_losses = []

        num = cls_prob.size(0)

        for idx in range(num):
            box_regression_ = box_regression[idx]
            # print(anchors_.size())
            cls_prob_ = cls_prob[idx]
            # obj * 4 
            # obj * 1 
            targets_ = targets[idx][:, :-1].data
            labels_ = targets[idx][:, -1].data.long()

            with torch.set_grad_enabled(False):
                box_localization = decode(loc=box_regression_, priors=anchors_, variances=self.variance)
                object_box_iou = jaccard(targets_, point_form(box_localization))
                t1 = self.bbox_threshold
                t2 = object_box_iou.max(dim=1, keepdim=True).values.clamp(min=t1 + 1e-12)
                # object_box_prob: P{a_{j} -> b_{i}}, shape: [i, j]
                object_box_prob = (
                    (object_box_iou - t1) / (t2 - t1)
                ).clamp(min=0, max=1)
                indices = torch.stack([torch.arange(len(labels_)).type_as(labels_), labels_], dim=0)

                # object_cls_box_prob: P{a_{j} -> b_{i}}, shape: [i, c, j]
                object_cls_box_prob = torch.sparse_coo_tensor(indices, object_box_prob)

                # image_box_iou: P{a_{j} \in A_{+}}, shape: [j, c]
                """
                from "start" to "end" implement:
                
                image_box_iou = torch.sparse.max(object_cls_box_prob, dim=0).t()
                
                """
                # start 
                indices = torch.nonzero(torch.sparse.sum(
                    object_cls_box_prob, dim=0
                ).to_dense()).t_()

                if indices.numel() == 0:
                    image_box_prob = torch.zeros(anchors_.size(0), self.num_classes).type_as(object_box_prob)
                else:
                    nonzero_box_prob = torch.where(
                        (labels_.unsqueeze(dim=-1) == indices[0]),
                        object_box_prob[:, indices[1]],
                        torch.tensor([0]).type_as(object_box_prob)
                    ).max(dim=0).values

                    image_box_prob = torch.sparse_coo_tensor(
                        indices.flip([0]), nonzero_box_prob,
                        size=(anchors_.size(0), self.num_classes)
                    ).to_dense()
                # end

                box_prob.append(image_box_prob)

            # construct bags for objects
            match_quality_matrix = jaccard(targets_, point_form(anchors_))
            _, matched = torch.topk(match_quality_matrix, self.pre_anchor_topk, dim=1, sorted=False)
            del match_quality_matrix
            # matched_cls_prob: P_{ij}^{cls}
            matched_cls_prob = torch.gather(
                cls_prob_[matched], 2, labels_.view(-1, 1, 1).repeat(1, self.pre_anchor_topk, 1)
            ).squeeze(2)

            # matched_box_prob: P_{ij}^{loc}
            # num_targets * topk * 4
            matched_object_targets = encode(matched=targets_.repeat(1, self.pre_anchor_topk).view(-1 ,4),
                                            priors=anchors_[matched].contiguous().view(-1 ,4), 
                                            variances=self.variance).view(-1, self.pre_anchor_topk, 4)
            retinanet_regression_loss = smooth_l1_loss(
                box_regression_[matched], matched_object_targets, *self.smooth_l1_loss_param
            )
            # objnum*topk
            matched_box_prob = torch.exp(-retinanet_regression_loss)
            # positive_losses: { -log( Mean-max(P_{ij}^{cls} * P_{ij}^{loc}) ) }
            positive_numels += targets_.size(0)
            positive_losses.append(self.positive_bag_loss_func(matched_cls_prob * matched_box_prob, dim=1))

        # positive_loss: \sum_{i}{ -log( Mean-max(P_{ij}^{cls} * P_{ij}^{loc}) ) } / ||B||
        positive_loss = torch.cat(positive_losses).sum() / max(1, positive_numels)

        # box_prob: P{a_{j} \in A_{+}}
        box_prob = torch.stack(box_prob, dim=0)

        negative_loss = self.negative_bag_loss_func(
            cls_prob * (1 - box_prob), self.focal_loss_gamma
        ) / max(1, positive_numels * self.pre_anchor_topk)

        # loss_n is loss_retina_negativeloss_retina_negative
        loss_p = positive_loss * self.focal_loss_alpha
        loss_n = negative_loss * (1 - self.focal_loss_alpha)
        # print("loss_p",loss_p)
        # print("loss_n",loss_n)

        return loss_p, loss_n
        
