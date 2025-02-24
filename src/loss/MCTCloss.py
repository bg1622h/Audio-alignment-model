import numpy as np, os, scipy
import torch
import torch.nn as nn
from itertools import groupby

"""
Taken from: https://github.com/christofw/multipitch_mctc/blob/main/libdl/nn_losses/mctc.py
"""
class mctc_we_loss(nn.CTCLoss):
    """ Multi-label Connectionist Temporal Classification (MCTC) Loss in "With Epsilon" (WE) encoding,
        i.e., there is an overall blank category, for which the probabilities of other components are ignored (epsilon)
        thus, other categories have components (blank, 1, [epsilon])

        Args:
        reduction='none'    No reduction / averaging applied to loss within this class.
                            Has to be done afterwards explicitly.

        For details see: https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#CTCLoss
        and
        C. Wigington, B.L. Price, S. Cohen: Multi-label Connectionist Temporal Classification. ICDAR 2019: 979-986

    """
    def __init__(self, reduction='none'):
        super(mctc_we_loss, self).__init__(reduction=reduction)
        assert reduction=='none', 'This loss is not tested with other reductions. Please apply reductions afterwards explicitly'

    def forward(self, log_probs, targets, input_lengths, target_lengths):

        ctc_loss = nn.CTCLoss(reduction=self.reduction)

        # Prepare targets
        print(targets.shape)
        print(log_probs.shape)
        char_unique, char_target = torch.unique(targets, dim=1, return_inverse=True)   # char_unique is the BatchCharacterList
        char_target = torch.remainder(char_target+1, char_unique.size(1)) # shift blank character to first position
        char_unique = torch.roll(char_unique, 1, -1)
        char_targ_condensed = torch.tensor([t[0] for t in groupby(char_target)][1:])
        target_torch = char_targ_condensed.unsqueeze(0) # no shift since blank_MCTC already exists on pos. 0

        # Overwrite target sequence length
        target_lengths = torch.tensor(target_torch.size(1), dtype=torch.long)

        # Prepare inputs
        input_logsoftmax = log_probs.unsqueeze(2)
        char_probs_nonblank = torch.matmul(1-char_unique[:, 1:].transpose(0, -1), torch.squeeze(input_logsoftmax[0, :, :, :])) \
        + torch.matmul(char_unique[:, 1:].transpose(0, -1), torch.squeeze(input_logsoftmax[1, :, :, :]))
        # recalculate first row (category blank) due to eps values (ignore other categories for computing blank probability)
        char_probs_blank = input_logsoftmax[1, :1, :, :].squeeze(2).squeeze(1)
        char_probs = torch.cat((char_probs_blank, char_probs_nonblank), dim=0)
        input_torch = char_probs.transpose(0, -1).unsqueeze(1)

        # Overwrite input sequence length
        input_lengths = torch.tensor(input_torch.size(0), dtype=torch.long)

        # Compute loss from characters
        mctc_loss = ctc_loss(input_torch, target_torch, input_lengths, target_lengths)

        return mctc_loss