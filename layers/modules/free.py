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

    def __init__(self, num_classes, overlap_thresh, use_gpu=True):
        super(FreeLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes # 20
        self.threshold = overlap_thresh# 0.5
        self.variance = cfg['variance'] # [0.1, 0.2]
        # 每个obj的正样本的数量
        self.pre_anchor_topk = 200 
        self.smooth_l1_loss_param = (0.75, 0.11)
        self.focal_loss_alpha = 0.5
        self.focal_loss_gamma = 2.0
        self.positive_bag_loss_func = positive_bag_loss
        self.negative_bag_loss_func = focal_loss

        # 新加的
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, predictions, targets):
        # 还是只有6个anchor  
        # loc_data batch*8732*4
        # conf_data batch*8732*20
        # priors 8732*4
        # 都代表预测值
        loc_data, conf_data, priors = predictions
        # print("loc_data",loc_data.size(),loc_data.max(),loc_data.min())
        # print("loc_data",loc_data.size(),loc_data.max(),loc_data.min())
        # print("conf_data",conf_data.size(),conf_data.max(),conf_data.min())
        # print("priors",priors.size(),priors.max(),priors.min())
        """
        print("start writing")
        with open('/home/outsider/Desktop/zbc_labor/results/ssd.txt', 'w') as f:
            f.write("box_cls:")
            f.write(str(conf_data))
            f.write('/n')
            f.write("box_regression:")
            f.write(str(loc_data))
        print("success writing")
        sys.exit(0)
        """
        # num是batch的大小
        # print("loc_data",loc_data.max(),loc_data.min())
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = priors.size(0)# 8732 是anchor的数量
        num_classes = self.num_classes # 20 是obj的数量
        # match priors (default boxes) and ground truth boxes
        # pred_loc_enco是已经encode的location结果 cx,cy,w,h
        pred_loc_enco = torch.Tensor(num, num_priors, 4)# 8*8732*4
        """
        t1,_ = torch.min(conf_data.view(-1, num_classes),0)
        t2,_ = torch.max(conf_data.view(-1, num_classes),0)
        cls_prob = -(conf_data.view(-1, num_classes)-t1)/(t2-t1)*7.5
        """
        # cls_prob batch*8732*20
        cls_prob = self.softmax(conf_data)
        # print("cls_prob",cls_prob[:, :, 0])   
        # cls_prob = torch.sigmoid(conf_data-3.8)
        # print("cls_prob",cls_prob.max(),cls_prob.min())
        box_prob=[]
        positive_numels = 0
        positive_losses = []
        """
        if self.use_gpu:
            pred_loc_enco = pred_loc_enco.cuda()
            cls_prob = cls_prob.cuda()
            box_prob = cls_prob.cuda()
            positive_losses = cls_prob.cuda()
            """
        for idx in range(num):
            cls_prob_ = cls_prob[idx]
            loc_data_ = loc_data[idx]
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data.long()
            # print("cls_prob_",cls_prob_.max(),cls_prob_.min())
            # print("loc_data_",loc_data_.max(),loc_data_.min())
            # 这里不需要更新参数,因为这里是评估anchor的prob方法
            with torch.set_grad_enabled(False):
                # pred_loc_enco是已经decode的anchor 的 location结果 cx,cy,w,h
                pred_loc_enco = decode(loc=loc_data_, priors=priors, variances=self.variance)
                # truth 代表的是实际obj的位置
                # labels 代表的是实际的obj的labels
                # anch_obj_iou计算的是prediction和truth的iou
                # freeanchor的loss是认为label等于原来的值减去1,不存在和背景匹配的说法
                
                # print("labels",labels.size())
                # print("lables",labels.max(),labels.min())
                # anch_obj_iou是anchor和obj的iou值,8732*obj_num
                anch_obj_iou = jaccard(point_form(pred_loc_enco), truths)
                anch_obj_iou = torch.transpose(anch_obj_iou, 1, 0)
                # print("anch_obj_iou",anch_obj_iou.size(),anch_obj_iou.max(),anch_obj_iou.min())
                # threhold 是每个object和anchor的iou值
                # start 到 end是计算saturated linear
                # start
                t1 = self.threshold
                # 每个object的最大匹配iou即是最好的匹配anchor
                t2 = anch_obj_iou.max(dim=1, keepdim=True).values.clamp(min=t1 + 1e-12)
                # 计算P{a_{j} -> b_{i}},代表第j个anchor与第i个object配合的概率
                
                anch_obj_iou = (
                    (anch_obj_iou - t1) / (t2 - t1)
                ).clamp(min=0, max=1)
                # num_obj*8732
                # print("anch_obj_iou",anch_obj_iou.size(),anch_obj_iou.max(),anch_obj_iou.min())
                
                # end
                # 扩展成object_cls_box labels*classnum*8732
                indices = torch.stack([torch.arange(len(labels)).type_as(labels), labels], dim=0)
                object_cls_box_prob = torch.sparse_coo_tensor(indices, anch_obj_iou)
                # image_box_prob就是每个anchor属于每个类的最大的IOU(j,c)
                # start
                # indices 是object_cls_box_prob的非零索引([[0,0,0],[0,0,0]...)
                indices = torch.nonzero(torch.sparse.sum(
                    object_cls_box_prob, dim=0
                ).to_dense()).t_()
                if indices.numel() == 0:
                    image_box_prob = torch.zeros(num_priors, self.num_classes).type_as(anch_obj_iou)
                else:
                    # print("indices[0]",indices[0])
                    #print("labels.unsqueeze(dim=-1)",labels.unsqueeze(dim=-1))
                    nonzero_box_prob = torch.where(
                        (labels.unsqueeze(dim=-1) == indices[0]),
                        anch_obj_iou[:, indices[1]],
                        torch.tensor([0]).type_as(anch_obj_iou)
                    ).max(dim=0).values
                    image_box_prob = torch.sparse_coo_tensor(
                        indices.flip([0]), nonzero_box_prob,
                        size=(num_priors, self.num_classes)
                    ).to_dense()
                    # print("image_box_prob",image_box_prob)
                # end
                box_prob.append(image_box_prob)
            # construct bags for objects
            # 找到每个obj的最好的50个匹配框架
            # matched是每个obj的前50个anchor的索引
            match_quality_matrix = jaccard(point_form(priors), truths)
            match_quality_matrix = torch.transpose(match_quality_matrix, 1, 0)
            # print("match_quality_matrix",match_quality_matrix.size(),match_quality_matrix.max(),match_quality_matrix.min())
            _, matched = torch.topk(match_quality_matrix, self.pre_anchor_topk, dim=1, sorted=False)
            del match_quality_matrix
            # print("matched",matched.size(),matched.max(),matched.min())
            # matched labels*50
            # matched_cls_prob: P_{ij}^{cls}    
            matched_cls_prob = torch.gather(
                cls_prob_[matched], 2, labels.view(-1, 1, 1).repeat(1, self.pre_anchor_topk, 1)
            ).squeeze(2)
            # print("matched_cls_prob",matched_cls_prob.size(),matched_cls_prob.max(),matched_cls_prob.min())
            # matched_box_prob: P_{ij}^{loc}
            # matched_object_targets是labelnum*topk*4
            matched_object_targets = encode(matched=truths.repeat(1, self.pre_anchor_topk, 1).view(-1 ,4),
                                             priors=priors[matched].contiguous().view(-1 ,4), 
                                             variances=self.variance).view(-1, self.pre_anchor_topk, 4)
            # smooth_l1_loss是比例的损失,因此两种定义是一样的
            retinanet_regression_loss = smooth_l1_loss(
                loc_data_[matched], matched_object_targets, *self.smooth_l1_loss_param
            )
            # matched_box_prob:obj_num*topk
            matched_box_prob = torch.exp(-retinanet_regression_loss)

             # positive_losses: { -log( Mean-max(P_{ij}^{cls} * P_{ij}^{loc}) ) }
            positive_numels += truths.size(0)
            positive_losses.append(self.positive_bag_loss_func(matched_cls_prob * matched_box_prob, dim=1))


        # positive_loss: \sum_{i}{ -log( Mean-max(P_{ij}^{cls} * P_{ij}^{loc}) ) } / ||B||
        positive_loss = torch.cat(positive_losses).sum() / max(1, positive_numels)
        
        # box_prob: P{a_{j} \in A_{+}}
        box_prob = torch.stack(box_prob, dim=0)
        # cls_prob是预测出来的每个anchor属于每个类的概率
        # box_prob是计算出来的每个anchor对于每个类的最大iou
        # negative_loss相当于计算  sum(-log(1-p))
        # P=每个anchor对于每个类预测属于该类的概率*每个anchor通过IOU计算得到的不属于该类的概率
        # negative_loss: \sum_{j}{ FL( (1 - P{a_{j} \in A_{+}}) * (1 - P_{j}^{bg}) ) } / n||B||
        
        """print("cls_prob max",cls_prob.max())

        print("cls_prob min",cls_prob.min())

        print("cls_prob size",cls_prob.size())
        """
        negative_loss = self.negative_bag_loss_func(
            cls_prob * (1 - box_prob), self.focal_loss_gamma
        ) / max(1, positive_numels * self.pre_anchor_topk)
        
        # loss_n is loss_retina_negativeloss_retina_negative
        loss_p = positive_loss * self.focal_loss_alpha
        loss_n = negative_loss * (1 - self.focal_loss_alpha)
        # # print("loss",loss_n+loss_p)
        return loss_p, loss_n
