import torch
import torch.nn as nn


class SASRecLoss(nn.Module):

    def __init__(
            self,
            positive_prefix,
            negative_prefix,
            output_prefix=None
    ):
        super().__init__()
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix
        self._output_prefix = output_prefix

    def forward(self, inputs):
        positive_scores = inputs[self._positive_prefix]  # (x)
        negative_scores = inputs[self._negative_prefix]  # (x)
        assert positive_scores.shape[0] == negative_scores.shape[0]

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            positive_scores, torch.ones_like(positive_scores)
        ) + torch.nn.functional.binary_cross_entropy_with_logits(
            negative_scores, torch.zeros_like(negative_scores)
        )

        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class IdentityLoss(nn.Module):

    def __init__(self, predictions_prefix, output_prefix=None):
        super().__init__()
        self._input_loss_key = predictions_prefix
        self._output_prefix = output_prefix

    def forward(self, inputs):
        loss = inputs[self._input_loss_key]
        assert loss.dim() == 0, "Loss must be a scalar tensor"
        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()
        return loss
