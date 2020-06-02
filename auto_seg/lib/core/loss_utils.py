from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import numpy
import torch
import torch.nn.functional as F


__all__ = ['map_get_pairs', 'log_det_by_cholesky']

def map_get_pairs(labels_4D, probs_4D, radius=3, is_combine=True):
    """get map pairs
    Args:
        labels_4D	:	labels, shape [N, C, H, W]
        probs_4D	:	probabilities, shape [N, C, H, W]
        radius		:	the square radius
    Return:
        tensor with shape [N, C, radius * radius, H - (radius - 1), W - (radius - 1)]
    """
    # pad to ensure the following slice operation is valid
    # pad_beg = int(radius // 2)
    # pad_end = radius - pad_beg

    # the original height and width
    label_shape = labels_4D.size()
    h, w = label_shape[2], label_shape[3]
    new_h, new_w = h - (radius - 1), w - (radius - 1)
    # https://pytorch.org/docs/stable/nn.html?highlight=f%20pad#torch.nn.functional.pad
    # padding = (pad_beg, pad_end, pad_beg, pad_end)
    # labels_4D, probs_4D = F.pad(labels_4D, padding), F.pad(probs_4D, padding)

    # get the neighbors
    la_ns = []
    pr_ns = []
    # for x in range(0, radius, 1):
    for y in range(0, radius, 1):
        for x in range(0, radius, 1):
            la_now = labels_4D[:, :, y:y + new_h, x:x + new_w]
            pr_now = probs_4D[:, :, y:y + new_h, x:x + new_w]
            la_ns.append(la_now)
            pr_ns.append(pr_now)

    if is_combine:
        # for calculating RMI
        pair_ns = la_ns + pr_ns
        p_vectors = torch.stack(pair_ns, dim=2)
        return p_vectors
    else:
        # for other purpose
        la_vectors = torch.stack(la_ns, dim=2)
        pr_vectors = torch.stack(pr_ns, dim=2)
        return la_vectors, pr_vectors



def log_det_by_cholesky(matrix):
    """
    Args:
        matrix: matrix must be a positive define matrix.
                shape [N, C, D, D].
    Ref:
        https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/ops/linalg/linalg_impl.py
    """
    # This uses the property that the log det(A) = 2 * sum(log(real(diag(C))))
    # where C is the cholesky decomposition of A.
    chol = torch.cholesky(matrix)
    # return 2.0 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-6), dim=-1)
    return 2.0 * torch.sum(torch.log(torch.diagonal(chol, dim1=-2, dim2=-1) + 1e-8), dim=-1)

