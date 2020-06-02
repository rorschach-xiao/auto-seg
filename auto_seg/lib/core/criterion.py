import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import logging

from config import config
from itertools import  filterfalse as ifilterfalse
from core.loss_utils import map_get_pairs,log_det_by_cholesky


_euler_num = 2.718281828				# 	euler number
_pi = 3.14159265						# 	pi
_ln_2_pi = 1.837877						# 	ln(2 * pi)
_CLIP_MIN = 1e-6            			# 	min clip value after softmax or sigmoid operations
_CLIP_MAX = 1.0    						# 	max clip value after softmax or sigmoid operations
_POS_ALPHA = 5e-4						# 	add this factor to ensure the AA^T is positive definite
_IS_SUM = 1                             # 	sum the loss per channel

'''

用于多分类任务的损失函数,具体包括：
  1.cross-entropy loss
  2.ohem cross-entropy loss
  3.lovasz-softmax
  4.RMI loss

'''

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label
        )

    def _forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

        loss = self.criterion(score, target)

        return loss

    def forward(self, score, target):

        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        weights = config.LOSS.BALANCE_WEIGHTS
        assert len(weights) == len(score)

        return sum([w * self._forward(x, target) for (w, x) in zip(weights, score)])


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ce_forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

        loss = self.criterion(score, target)

        return loss

    def _ohem_forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target):

        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        weights = config.LOSS.BALANCE_WEIGHTS
        assert len(weights) == len(score)
        if config.TEST.OUTPUT_INDEX==-1:
            functions = [self._ce_forward] * \
                (len(weights) - 1) + [self._ohem_forward]
        else:
            functions = [self._ce_forward] * \
                        (len(weights) - 1)
            functions.insert(config.TEST.OUTPUT_INDEX,self._ohem_forward)
        return sum([
            w * func(x, target)
            for (w, x, func) in zip(weights, score, functions)
        ])

class LovaszSoftmax(nn.Module):
    def __init__(self,ignore_label=-1,weight = None,per_image=True):
        super(LovaszSoftmax,self).__init__()
        self.per_image = per_image
        self.weight = weight
        self.ignore_label=ignore_label

    def _forward(self, score,target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
        return self.lovasz_softmax(score, target, classes='present',per_image=self.per_image,ignore=self.ignore_label)

    def forward(self, score,target) :
        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        weights = config.LOSS.BALANCE_WEIGHTS
        assert len(weights) == len(score)

        return sum([w * self._forward(x, target) for (w, x) in zip(weights, score)])

    def lovasz_softmax(self,probas, labels, classes='present', per_image=False, ignore=None):
        """
        Multi-class Lovasz-Softmax loss
          probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
                  Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
          labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
          classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
          per_image: compute the loss per image instead of per batch
          ignore: void class labels
        """
        probas = F.softmax(probas,dim=1)
        if per_image:
            loss = self.mean(
                self.lovasz_softmax_flat(*self.flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                for prob, lab in zip(probas, labels))
        else:
            loss = self.lovasz_softmax_flat(*self.flatten_probas(probas, labels, ignore), classes=classes)
        return loss

    def lovasz_grad(self,gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def lovasz_softmax_flat(self,probas, labels, classes='present'):
        """
        Multi-class Lovasz-Softmax loss
          probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
          labels: [P] Tensor, ground truth labels (between 0 and C - 1)
          classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        """
        if probas.numel() == 0:
            # only void pixels, the gradients should be 0
            return probas * 0.
        C = probas.size(1)
        losses = []
        class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
        for c in class_to_sum:
            fg = (labels == c).float()  # foreground for class c
            if (classes is 'present' and fg.sum() == 0):
                continue
            if C == 1:
                if len(classes) > 1:
                    raise ValueError('Sigmoid output possible only with 1 class')
                class_pred = probas[:, 0]
            else:
                class_pred = probas[:, c]
            errors = (Variable(fg) - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, Variable(self.lovasz_grad(fg_sorted))))
        return mean(losses)

    def flatten_probas(self,probas, labels, ignore=None):
        """
        Flattens predictions in the batch
        """
        if probas.dim() == 3:
            # assumes output of a sigmoid layer
            B, H, W = probas.size()
            probas = probas.view(B, 1, H, W)
        B, C, H, W = probas.size()
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        labels = labels.view(-1)
        if ignore is None:
            return probas, labels
        valid = (labels != ignore)
        vprobas = probas[valid.nonzero().squeeze()]
        vlabels = labels[valid]
        return vprobas, vlabels

class RMILoss(nn.Module):
    def __init__(self,
                 num_classes=21,
                 rmi_radius=3,
                 rmi_pool_way=0,
                 rmi_pool_size=3,
                 rmi_pool_stride=3,
                 loss_weight_lambda=0.5,
                 lambda_way=1):
        super(RMILoss, self).__init__()
        self.num_classes = num_classes
        # radius choices
        assert rmi_radius in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.rmi_radius = rmi_radius
        assert rmi_pool_way in [0, 1, 2, 3]
        self.rmi_pool_way = rmi_pool_way

        # set the pool_size = rmi_pool_stride
        assert rmi_pool_size == rmi_pool_stride
        self.rmi_pool_size = rmi_pool_size
        self.rmi_pool_stride = rmi_pool_stride
        self.weight_lambda = loss_weight_lambda
        self.lambda_way = lambda_way

        # dimension of the distribution
        self.half_d = self.rmi_radius * self.rmi_radius
        self.d = 2 * self.half_d
        self.kernel_padding = self.rmi_pool_size // 2
        # ignore class
        self.ignore_index = 255
    def _forward(self,score,target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
        if self.num_classes>2:
            return self.forward_softmax_sigmoid(score,target)
        else:
            return self.forward_sigmoid(score,target)


    def forward(self, score,target) :
        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        weights = config.LOSS.BALANCE_WEIGHTS
        assert len(weights) == len(score)

        return sum([w * self._forward(x, target) for (w, x) in zip(weights, score)])

    def forward_softmax_sigmoid(self, logits_4D, labels_4D):
        """
        Using both softmax and sigmoid operations.
        Args:
            logits_4D 	:	[N, C, H, W], dtype=float32
            labels_4D 	:	[N, H, W], dtype=long
        """
        # PART I -- get the normal cross entropy loss
        normal_loss = F.cross_entropy(input=logits_4D,
                                      target=labels_4D.long(),
                                      ignore_index=self.ignore_index,
                                      reduction='mean')

        # PART II -- get the lower bound of the region mutual information
        # get the valid label and logits
        # valid label, [N, C, H, W]
        label_mask_3D = labels_4D < self.num_classes
        valid_onehot_labels_4D = F.one_hot(labels_4D.long() * label_mask_3D.long(),
                                           num_classes=self.num_classes).float()
        label_mask_3D = label_mask_3D.float()
        valid_onehot_labels_4D = valid_onehot_labels_4D * label_mask_3D.unsqueeze(dim=3)
        valid_onehot_labels_4D = valid_onehot_labels_4D.permute(0, 3, 1, 2).requires_grad_(False)
        # valid probs
        probs_4D = F.sigmoid(logits_4D) * label_mask_3D.unsqueeze(dim=1)
        probs_4D = probs_4D.clamp(min=_CLIP_MIN, max=_CLIP_MAX)

        # get region mutual information
        rmi_loss = self.rmi_lower_bound(valid_onehot_labels_4D, probs_4D)

        # add together
        final_loss = (self.weight_lambda * normal_loss + rmi_loss * (1 - self.weight_lambda) if self.lambda_way
                      else normal_loss + rmi_loss * self.weight_lambda)

        return final_loss

    def forward_sigmoid(self, logits_4D, labels_4D):
        """
        Using the sigmiod operation both.
        Args:
            logits_4D 	:	[N, C, H, W], dtype=float32
            labels_4D 	:	[N, H, W], dtype=long
        """
        # label mask -- [N, H, W, 1]
        label_mask_3D = labels_4D < self.num_classes

        # valid label
        valid_onehot_labels_4D = F.one_hot(labels_4D.long() * label_mask_3D.long(),
                                           num_classes=self.num_classes).float()
        label_mask_3D = label_mask_3D.float()
        label_mask_flat = label_mask_3D.view([-1, ])
        valid_onehot_labels_4D = valid_onehot_labels_4D * label_mask_3D.unsqueeze(dim=3)
        valid_onehot_labels_4D.requires_grad_(False)

        # PART I -- calculate the sigmoid binary cross entropy loss
        valid_onehot_label_flat = valid_onehot_labels_4D.view([-1, self.num_classes]).requires_grad_(False)
        logits_flat = logits_4D.permute(0, 2, 3, 1).contiguous().view([-1, self.num_classes])

        # binary loss, multiplied by the not_ignore_mask
        valid_pixels = torch.sum(label_mask_flat)
        binary_loss = F.binary_cross_entropy_with_logits(logits_flat,
                                                         target=valid_onehot_label_flat,
                                                         weight=label_mask_flat.unsqueeze(dim=1),
                                                         reduction='sum')
        bce_loss = torch.div(binary_loss, valid_pixels + 1.0)

        # PART II -- get rmi loss
        # onehot_labels_4D -- [N, C, H, W]
        probs_4D = logits_4D.sigmoid() * label_mask_3D.unsqueeze(dim=1) + _CLIP_MIN
        valid_onehot_labels_4D = valid_onehot_labels_4D.permute(0, 3, 1, 2).requires_grad_(False)

        # get region mutual information
        rmi_loss = self.rmi_lower_bound(valid_onehot_labels_4D, probs_4D)

        # add together
        final_loss = (self.weight_lambda * bce_loss + rmi_loss * (1 - self.weight_lambda) if self.lambda_way
                      else bce_loss + rmi_loss * self.weight_lambda)

        return final_loss

    def rmi_lower_bound(self, labels_4D, probs_4D):
        """
        calculate the lower bound of the region mutual information.
        Args:
            labels_4D 	:	[N, C, H, W], dtype=float32
            probs_4D 	:	[N, C, H, W], dtype=float32
        """
        assert labels_4D.size() == probs_4D.size()

        p, s = self.rmi_pool_size, self.rmi_pool_stride
        if self.rmi_pool_stride > 1:
            if self.rmi_pool_way == 0:
                labels_4D = F.max_pool2d(labels_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
                probs_4D = F.max_pool2d(probs_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
            elif self.rmi_pool_way == 1:
                labels_4D = F.avg_pool2d(labels_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
                probs_4D = F.avg_pool2d(probs_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
            elif self.rmi_pool_way == 2:
                # interpolation
                shape = labels_4D.size()
                new_h, new_w = shape[2] // s, shape[3] // s
                labels_4D = F.interpolate(labels_4D, size=(new_h, new_w), mode='nearest')
                probs_4D = F.interpolate(probs_4D, size=(new_h, new_w), mode='bilinear', align_corners=True)
            else:
                raise NotImplementedError("Pool way of RMI is not defined!")
        # we do not need the gradient of label.
        label_shape = labels_4D.size()
        n, c = label_shape[0], label_shape[1]

        # combine the high dimension points from label and probability map. new shape [N, C, radius * radius, H, W]
        la_vectors, pr_vectors = map_get_pairs(labels_4D, probs_4D, radius=self.rmi_radius, is_combine=0)

        la_vectors = la_vectors.view([n, c, self.half_d, -1]).type(torch.cuda.DoubleTensor).requires_grad_(False)
        pr_vectors = pr_vectors.view([n, c, self.half_d, -1]).type(torch.cuda.DoubleTensor)

        # small diagonal matrix, shape = [1, 1, radius * radius, radius * radius]
        diag_matrix = torch.eye(self.half_d).unsqueeze(dim=0).unsqueeze(dim=0)

        # the mean and covariance of these high dimension points
        # Var(X) = E(X^2) - E(X) E(X), N * Var(X) = X^2 - X E(X)
        la_vectors = la_vectors - la_vectors.mean(dim=3, keepdim=True)
        la_cov = torch.matmul(la_vectors, la_vectors.transpose(2, 3))

        pr_vectors = pr_vectors - pr_vectors.mean(dim=3, keepdim=True)
        pr_cov = torch.matmul(pr_vectors, pr_vectors.transpose(2, 3))
        # https://github.com/pytorch/pytorch/issues/7500
        # waiting for batched torch.cholesky_inverse()
        pr_cov_inv = torch.inverse(pr_cov + diag_matrix.type_as(pr_cov) * _POS_ALPHA)
        # if the dimension of the point is less than 9, you can use the below function
        # to acceleration computational speed.
        # pr_cov_inv = utils.batch_cholesky_inverse(pr_cov + diag_matrix.type_as(pr_cov) * _POS_ALPHA)

        la_pr_cov = torch.matmul(la_vectors, pr_vectors.transpose(2, 3))
        # the approxiamation of the variance, det(c A) = c^n det(A), A is in n x n shape;
        # then log det(c A) = n log(c) + log det(A).
        # appro_var = appro_var / n_points, we do not divide the appro_var by number of points here,
        # and the purpose is to avoid underflow issue.
        # If A = A^T, A^-1 = (A^-1)^T.
        appro_var = la_cov - torch.matmul(la_pr_cov.matmul(pr_cov_inv), la_pr_cov.transpose(-2, -1))
        # appro_var = la_cov - torch.chain_matmul(la_pr_cov, pr_cov_inv, la_pr_cov.transpose(-2, -1))
        # appro_var = torch.div(appro_var, n_points.type_as(appro_var)) + diag_matrix.type_as(appro_var) * 1e-6

        # The lower bound. If A is nonsingular, ln( det(A) ) = Tr( ln(A) ).
        rmi_now = 0.5 * log_det_by_cholesky(appro_var + diag_matrix.type_as(appro_var) * _POS_ALPHA)
        # rmi_now = 0.5 * torch.logdet(appro_var + diag_matrix.type_as(appro_var) * _POS_ALPHA)

        # mean over N samples. sum over classes.
        rmi_per_class = rmi_now.view([-1, self.num_classes]).mean(dim=0).float()
        # is_half = False
        # if is_half:
        #	rmi_per_class = torch.div(rmi_per_class, float(self.half_d / 2.0))
        # else:
        rmi_per_class = torch.div(rmi_per_class, float(self.half_d))

        rmi_loss = torch.sum(rmi_per_class) if _IS_SUM else torch.mean(rmi_per_class)
        return rmi_loss



'''

用于二分类任务的损失函数,具体包括：
  1.binaary cross-entropy loss
  2.soft-dice loss
  3.combo loss
  4.lovasz-sigmoid loss
  5.lovasz-hinge loss
  
'''

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, per_image=True):
        super(SoftDiceLoss,self).__init__()
        self.size_average = size_average
        #self.register_buffer('weight', weight)
        self.per_image = per_image
        self.weights = weight

    def soft_dice_loss(self,outputs, targets, weights=None, per_image=True, per_channel=False):
        batch_size, n_channels = outputs.size(0), outputs.size(1)

        eps = 1e-6
        n_parts = 1
        if per_image:
            n_parts = batch_size
        if per_channel:
            n_parts = batch_size * n_channels
        # use softmax on output channel when channel is 2
        if config.DATASET.NUM_CLASSES == 2:
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            outputs_fg = outputs[:,1,:,:]
            outputs_bg = outputs[:,0,:,:]
        # use sigmoid on output channel when channel is 1
        else :
            outputs = torch.sigmoid(outputs[:,0,:,:]).float()
            outputs_fg = outputs
            outputs_bg = 1.- outputs

        # caculate foreground dice
        dice_target = targets.contiguous().view(n_parts, -1).float()
        dice_output = outputs_fg.contiguous().view(n_parts, -1)
        intersection = torch.sum(dice_output * dice_target, dim=1)
        union = (torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + eps)

        # caculate background dice
        dice_output_bg = outputs_bg.contiguous().view(n_parts, -1)
        dice_target_bg = torch.where(dice_target == 0, torch.full_like(dice_target, 1),
                                     torch.full_like(dice_target, 0))
        intersection_bg = torch.sum(dice_output_bg * dice_target_bg, dim=1)
        union_bg = torch.sum(dice_output_bg, dim=1) + torch.sum(dice_target_bg, dim=1) + eps

        if (weights is not None):
            cuda_id = intersection.device.index
            loss = (weights[0].cuda(cuda_id) * (1 - (intersection + eps) / union) + weights[1].cuda(cuda_id) * (
                        1 - (intersection_bg + eps) / union_bg)).float().mean()
        else:
            loss = (1 - (intersection + eps) / union + (1 - (intersection_bg + eps) / union_bg)).float().mean()

        return loss

    def _forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=False)
        return self.soft_dice_loss(score, target, weights=self.weights,per_image=self.per_image)
    def forward(self, score,target) :
        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        weights = config.LOSS.BALANCE_WEIGHTS
        assert len(weights) == len(score)

        return sum([w * self._forward(x, target) for (w, x) in zip(weights, score)])

class LovaszHinge(nn.Module):
    def __init__(self,ignore_label=255,weight = None,per_image=True):
        super(LovaszHinge,self).__init__()
        self.per_image = per_image
        self.weight = weight
        self.ignore_label=ignore_label

    def _forward(self, score,target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=False)
        score = score[:,0,:,:]
        return self.symmetric_lovasz(score,target)

    def forward(self, score,target) :
        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        weights = config.LOSS.BALANCE_WEIGHTS
        assert len(weights) == len(score)

        return sum([w * self._forward(x, target) for (w, x) in zip(weights, score)])


    def symmetric_lovasz(self,outputs, targets,):
        return (self.lovasz_hinge(outputs, targets,per_image=self.per_image,ignore=self.ignore_label) +
                self.lovasz_hinge(-outputs, 1 - targets,per_image=self.per_image,ignore=self.ignore_label)) / 2

    def lovasz_hinge(self,logits, labels, per_image=True, ignore=None):
        """
        Binary Lovasz hinge loss
          logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
          labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
          per_image: compute the loss per image instead of per batch
          ignore: void class id
        """
        if per_image:
            loss = mean(self.lovasz_hinge_flat(*self.flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                        for log, lab in zip(logits, labels))
        else:
            loss = self.lovasz_hinge_flat(*self.flatten_binary_scores(logits, labels, ignore))
        return loss
    def lovasz_grad(self,gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def lovasz_hinge_flat(self,logits, labels):
        """
        Binary Lovasz hinge loss
          logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
          labels: [P] Tensor, binary ground truth labels (0 or 1)
          ignore: label to ignore
        """
        if len(labels) == 0:
            # only void pixels, the gradients should be 0
            return logits.sum() * 0.
        signs = 2. * labels.float() - 1.
        errors = (1. - logits * Variable(signs))
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        grad = self.lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), Variable(grad))
        return loss

    def flatten_binary_scores(self,scores, labels, ignore=None):
        """
        Flattens predictions in the batch (binary case)
        Remove labels equal to 'ignore'
        """
        scores = scores.view(-1)
        labels = labels.view(-1)
        if ignore is None:
            return scores, labels
        valid = (labels != ignore)
        vscores = scores[valid]
        vlabels = labels[valid]
        return vscores, vlabels

class LovaszSigmoid(LovaszSoftmax):
    def __init__(self,ignore_label=255,weight = None,per_image=True):
        super(LovaszSigmoid,self).__init__()
        self.per_image = per_image
        self.weight = weight
        self.ignore_label=ignore_label
    def _forward(self, score,target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(
                h, w), mode='bilinear', align_corners=False)
        score = torch.sigmoid(score[:,0,:,:])
        return self.lovasz_softmax(probas=score,labels=target,classes=[1],per_image=True)
    def forward(self, score,target) :
        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        weights = config.LOSS.BALANCE_WEIGHTS
        assert len(weights) == len(score)

        return sum([w * self._forward(x, target) for (w, x) in zip(weights, score)])

class StableBCELoss(torch.nn.modules.Module):
    def __init__(self,ignore_label=255,weight = None):
         super(StableBCELoss, self).__init__()
         self.weights = weight
         self.ignore = ignore_label
    def _forward(self, inputs, target,weight=None):
         """
           Binary Cross entropy loss
             logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
             labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
             ignore: void class id
         """
         ph, pw = inputs.size(2), inputs.size(3)
         h, w = target.size(1), target.size(2)
         if ph != h or pw != w:
             inputs = F.interpolate(input=inputs, size=(
                 h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)
         inputs = inputs[:,0,:,:].float()
         inputs, target = self.flatten_binary_scores(inputs, target, self.ignore)
         target = Variable(target.float())
         neg_abs = - inputs.abs()
         loss = inputs.clamp(min=0) - inputs * target + (1 + neg_abs.exp()).log()

         return loss.mean()
    def flatten_binary_scores(self,scores, labels, ignore=None):
        """
        Flattens predictions in the batch (binary case)
        Remove labels equal to 'ignore'
        """
        scores = scores.view(-1)
        labels = labels.view(-1)
        if ignore is None:
            return scores, labels
        valid = (labels != ignore)
        vscores = scores[valid]
        vlabels = labels[valid]
        return vscores, vlabels
    def forward(self, score,target) :
        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        weights = config.LOSS.BALANCE_WEIGHTS
        assert len(weights) == len(score)

        return sum([w * self._forward(x, target) for (w, x) in zip(weights, score)])

class ComboLoss(nn.Module):
    def __init__(self,ignore_label=255,channel_weight = [0.5,1],data_weight=None):
         super(ComboLoss, self).__init__()
         self.channel_weights = channel_weight
         self.weights = data_weight
         self.ignore = ignore_label
    def forward(self, inputs, target):
        softdice_loss = SoftDiceLoss(weight = self.weights)(inputs,Variable(target))
        bce_loss = StableBCELoss(ignore_label=self.ignore)(inputs,Variable(target))
        combo_loss = self.channel_weights[0]*softdice_loss+self.channel_weights[1]*bce_loss
        return combo_loss


'''
一些辅助函数

'''

def isnan(x):
    return x != x

def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

def iou_dice_binary(preds, labels, EMPTY=1., ignore=255, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    dices = []
    pixel_accs = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        pa = (label == pred.long()).sum().float()/(label!=ignore).sum().float()
        if not union:
            iou = EMPTY
            dice = EMPTY
        else:
            iou = float(intersection) / float(union)
            dice = 2*float(intersection) / (float(union)+float(intersection))
        pixel_accs.append(pa)
        dices.append(dice)
        ious.append(iou)
    iou = mean(ious)  # mean accross images if per_image
    dice = mean(dices)
    pixel_acc = mean(pixel_accs)
    return iou,dice,pixel_acc


