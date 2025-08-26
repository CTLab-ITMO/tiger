import copy

from utils import MetaParent, get_activation_function, maybe_to_list, DEVICE

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLoss(metaclass=MetaParent):
    pass


class TorchLoss(BaseLoss, nn.Module):
    pass


class IdentityLoss(BaseLoss, config_name='identity'):

    def __call__(self, inputs):
        return inputs


class CompositeLoss(TorchLoss, config_name='composite'):

    def __init__(self, losses, weights=None, output_prefix=None):
        super().__init__()
        self._losses = losses
        self._weights = weights or [1.0] * len(losses)
        self._output_prefix = output_prefix

    @classmethod
    def create_from_config(cls, config, **kwargs):
        losses = []
        weights = []

        for loss_cfg in copy.deepcopy(config)['losses']:
            weight = loss_cfg.pop('weight') if 'weight' in loss_cfg else 1.0
            loss_function = BaseLoss.create_from_config(loss_cfg)

            weights.append(weight)
            losses.append(loss_function)

        return cls(losses=losses, weights=weights, output_prefix=config.get('output_prefix'))

    def forward(self, inputs):
        total_loss = 0.0
        for loss, weight in zip(self._losses, self._weights):
            total_loss += weight * loss(inputs)

        if self._output_prefix is not None:
            inputs[self._output_prefix] = total_loss.cpu().item()

        return total_loss

class SampleLogSoftmaxLoss(TorchLoss, config_name="sample_logsoftmax"):
    def __init__(self, predictions_prefix, labels):
        super().__init__()
        self._predictions_prefix = predictions_prefix
        self._labels = labels

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            predictions_prefix=config.get("predictions_prefix"),
            labels=config.get("labels"),
        )

    def forward(self, inputs):  # use log soft max
        logits = inputs[self._predictions_prefix]
        candidates = inputs[self._labels]

        assert len(logits.shape) in [2, 3]

        batch_size = logits.shape[0]
        seq_len = logits.shape[1]

        if len(logits.shape) == 3:
            loss = -torch.gather(
                torch.log_softmax(logits, dim=-1).reshape(
                    batch_size * seq_len, logits.shape[-1]
                ),
                dim=-1,
                index=candidates.reshape(batch_size * seq_len, 1),
            ).mean()
        else:
            loss = -torch.gather(
                torch.log_softmax(logits, dim=-1),
                dim=-1,
                index=candidates.reshape(batch_size, 1),
            ).mean()

        return loss


class BatchLogSoftmaxLoss(TorchLoss, config_name='batch_logsoftmax'):

    def __init__(self, predictions_prefix, candidates_prefix):
        super().__init__()
        self._predictions_prefix = predictions_prefix
        self._candidates_prefix = candidates_prefix

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            predictions_prefix=config.get('predictions_prefix'),
            candidates_prefix=config.get('candidates_prefix')
        )

    def forward(self, inputs):  # use log soft max
        predictions = inputs[self._predictions_prefix]
        candidates = inputs[self._candidates_prefix]

        dot_product_matrix = predictions @ candidates.T

        row_log_softmax = nn.LogSoftmax(dim=1)
        softmax_matrix = -row_log_softmax(dot_product_matrix)

        diagonal_elements = torch.diag(softmax_matrix)

        loss = diagonal_elements.mean()

        return loss


class CrossEntropyLoss(TorchLoss, config_name='ce'):

    def __init__(self, predictions_prefix, labels_prefix, output_prefix=None):
        super().__init__()
        self._pred_prefix = predictions_prefix
        self._labels_prefix = labels_prefix
        self._output_prefix = output_prefix

        self._loss = nn.CrossEntropyLoss()

    def forward(self, inputs):
        all_logits = inputs[self._pred_prefix]  # (all_items, num_classes)
        all_labels = inputs['{}.ids'.format(self._labels_prefix)]  # (all_items)
        assert all_logits.shape[0] == all_labels.shape[0]

        loss = self._loss(all_logits, all_labels)  # (1)
        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss

class RqVaeLoss(TorchLoss, config_name="rqvae_loss"):
    def __init__(self, beta, output_prefix=None):
        super().__init__()
        self.beta = beta
        self._output_prefix = output_prefix

        self._loss = nn.MSELoss()

    @classmethod
    def create_from_config(cls, config, **kwargs):
        # 0.25 is default Beta in paper
        return cls(
            beta=config.get("beta", 0.25),
            output_prefix=config["output_prefix"],
        )

    def forward(self, inputs):
        embeddings = inputs["embeddings"]
        embeddings_restored = inputs["embeddings_restored"]
        remainders = inputs["remainders"]
        codebooks_vectors = inputs["codebooks_vectors"]

        rqvae_loss = 0

        for remainder, codebook_vectors in zip(remainders, codebooks_vectors):
            rqvae_loss += self.beta * self._loss(remainder, codebook_vectors.detach())
            rqvae_loss += self._loss(codebook_vectors, remainder.detach())

        recon_loss = self._loss(embeddings_restored, embeddings)
        loss = (recon_loss + rqvae_loss).mean(dim=0)

        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class BinaryCrossEntropyLoss(TorchLoss, config_name='bce'):

    def __init__(
            self,
            predictions_prefix,
            labels_prefix,
            with_logits=True,
            output_prefix=None
    ):
        super().__init__()
        self._pred_prefix = predictions_prefix
        self._labels_prefix = labels_prefix
        self._output_prefix = output_prefix

        if with_logits:
            self._loss = nn.BCEWithLogitsLoss()
        else:
            self._loss = nn.BCELoss()

    def forward(self, inputs):
        all_logits = inputs[self._pred_prefix].float()  # (all_batch_items)
        all_labels = inputs[self._labels_prefix].float()  # (all_batch_items)
        assert all_logits.shape[0] == all_labels.shape[0]

        loss = self._loss(all_logits, all_labels)  # (1)
        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class BPRLoss(TorchLoss, config_name='bpr'):

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
        pos_scores = inputs[self._positive_prefix]  # (all_batch_items)
        neg_scores = inputs[self._negative_prefix]  # (all_batch_items)
        loss = -torch.log((pos_scores - neg_scores).sigmoid() + 1e-9).mean()  # (1)

        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class RegularizationLoss(TorchLoss, config_name='regularization'):

    def __init__(self, prefix, output_prefix=None):
        super().__init__()
        self._prefix = maybe_to_list(prefix)
        self._output_prefix = output_prefix

    def forward(self, inputs):
        loss = 0.0
        for prefix in self._prefix:
            loss += (1/2) * inputs[prefix].pow(2).mean()

        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class FpsLoss(TorchLoss, config_name='fps'):

    def __init__(
            self,
            fst_embeddings_prefix,
            snd_embeddings_prefix,
            tau=1.0,
            normalize_embeddings=False,
            use_mean=True,
            output_prefix=None
    ):
        super().__init__()
        self._fst_embeddings_prefix = fst_embeddings_prefix
        self._snd_embeddings_prefix = snd_embeddings_prefix
        self._tau = tau
        self._loss_function = nn.CrossEntropyLoss(reduction='mean' if use_mean else 'sum')
        self._normalize_embeddings = normalize_embeddings
        self._output_prefix = output_prefix

    def forward(self, inputs):
        fst_embeddings = inputs[self._fst_embeddings_prefix]  # (x, embedding_dim)
        snd_embeddings = inputs[self._snd_embeddings_prefix]  # (x, embedding_dim)

        batch_size = fst_embeddings.shape[0]

        combined_embeddings = torch.cat((fst_embeddings, snd_embeddings), dim=0)  # (2 * x, embedding_dim)

        if self._normalize_embeddings:
            combined_embeddings = torch.nn.functional.normalize(
                combined_embeddings, p=2, dim=-1, eps=1e-6
            )  # (2 * x, embedding_dim)

        similarity_scores = torch.mm(
            combined_embeddings,
            combined_embeddings.T
        ) / self._tau  # (2 * x, 2 * x)

        positive_samples = torch.cat(
            (torch.diag(similarity_scores, batch_size), torch.diag(similarity_scores, -batch_size)),
            dim=0
        ).reshape(2 * batch_size, 1)  # (2 * x, 1)
        assert torch.allclose(torch.diag(similarity_scores, batch_size), torch.diag(similarity_scores, -batch_size))

        mask = torch.ones(2 * batch_size, 2 * batch_size, dtype=torch.bool)  # (2 * x, 2 * x)
        mask = mask.fill_diagonal_(0)  # Remove equal embeddings scores
        for i in range(batch_size):  # Remove positives
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0

        negative_samples = similarity_scores[mask].reshape(2 * batch_size, -1)  # (2 * x, 2 * x - 2)

        labels = torch.zeros(2 * batch_size).to(positive_samples.device).long()  # (2 * x)
        logits = torch.cat((positive_samples, negative_samples), dim=1)  # (2 * x, 2 * x - 1)

        loss = self._loss_function(logits, labels) / 2  # (1)

        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class SASRecLoss(TorchLoss, config_name='sasrec'):

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


class SamplesSoftmaxLoss(TorchLoss, config_name='sampled_softmax'):

    def __init__(
            self,
            queries_prefix,
            positive_prefix,
            negative_prefix,
            output_prefix=None
    ):
        super().__init__()
        self._queries_prefix = queries_prefix
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix
        self._output_prefix = output_prefix

    def forward(self, inputs):
        queries_embeddings = inputs[self._queries_prefix]  # (batch_size, embedding_dim)
        positive_embeddings = inputs[self._positive_prefix]  # (batch_size, embedding_dim)
        negative_embeddings = inputs[self._negative_prefix]  # (num_negatives, embedding_dim) or (batch_size, num_negatives, embedding_dim)

        # b -- batch_size, d -- embedding_dim
        positive_scores = torch.einsum(
            'bd,bd->b',
            queries_embeddings,
            positive_embeddings
        ).unsqueeze(-1)  # (batch_size, 1)

        if negative_embeddings.dim() == 2:  # (num_negatives, embedding_dim)
            # b -- batch_size, n -- num_negatives, d -- embedding_dim
            negative_scores = torch.einsum(
                'bd,nd->bn',
                queries_embeddings,
                negative_embeddings
            )  # (batch_size, num_negatives)
        else:
            assert negative_embeddings.dim() == 3  # (batch_size, num_negatives, embedding_dim)
            # b -- batch_size, n -- num_negatives, d -- embedding_dim
            negative_scores = torch.einsum(
                'bd,bnd->bn',
                queries_embeddings,
                negative_embeddings
            )  # (batch_size, num_negatives)
        all_scores = torch.cat([positive_scores, negative_scores], dim=1)  # (batch_size, 1 + num_negatives)

        logits = torch.log_softmax(all_scores, dim=1)  # (batch_size, 1 + num_negatives)
        loss = (-logits)[:, 0]  # (batch_size)
        loss = loss.mean()  # (1)

        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class S3RecPretrainLoss(TorchLoss, config_name='s3rec_pretrain'):

    def __init__(
            self,
            positive_prefix,
            negative_prefix,
            representation_prefix,
            output_prefix=None
    ):
        super().__init__()
        self._positive_prefix = positive_prefix
        self._negative_prefix = negative_prefix
        self._representation_prefix = representation_prefix
        self._criterion = nn.BCEWithLogitsLoss(reduction="none")
        self._output_prefix = output_prefix

    def forward(self, inputs):
        positive_embeddings = inputs[self._positive_prefix]  # (x, embedding_dim)
        negative_embeddings = inputs[self._negative_prefix]  # (x, embedding_dim)
        current_embeddings = inputs[self._representation_prefix]  # (x, embedding_dim)
        assert positive_embeddings.shape[0] == negative_embeddings.shape[0] == current_embeddings.shape[0]

        positive_scores = torch.einsum(
            'bd,bd->b',
            positive_embeddings,
            current_embeddings
        )  # (x)

        negative_scores = torch.einsum(
            'bd,bd->b',
            negative_embeddings,
            current_embeddings
        )  # (x)

        distance = torch.sigmoid(positive_scores) - torch.sigmoid(negative_scores)  # (x)
        loss = torch.sum(self._criterion(distance, torch.ones_like(distance, dtype=torch.float32)))  # (1)
        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class Cl4sRecLoss(TorchLoss, config_name='cl4srec'):

    def __init__(
            self,
            current_representation,
            all_items_representation,
            tau=1.0,
            output_prefix=None
    ):
        super().__init__()
        self._current_representation = current_representation
        self._all_items_representation = all_items_representation
        self._loss_function = nn.CrossEntropyLoss()
        self._tau = tau
        self._output_prefix = output_prefix

    def forward(self, inputs):
        current_representation = inputs[self._current_representation]  # (batch_size, embedding_dim)
        all_items_representation = inputs[
            self._all_items_representation
        ]  # (batch_size, num_negatives + 1, embedding_dim)

        batch_size = current_representation.shape[0]

        logits = torch.einsum(
            'bnd,bd->bn',
            all_items_representation,
            current_representation
        )  # (batch_size, num_negatives + 1)
        labels = logits.new_zeros(batch_size)  # (batch_size)

        loss = self._loss_function(logits, labels)

        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class DuorecSSLLoss(TorchLoss, config_name='duorec_ssl'):

    def __init__(
            self,
            original_embedding_prefix,
            dropout_embedding_prefix,
            similar_embedding_prefix,
            normalize_embeddings=False,
            tau=1.0,
            output_prefix=None
    ):
        super().__init__()
        self._original_embedding_prefix = original_embedding_prefix
        self._dropout_embedding_prefix = dropout_embedding_prefix
        self._similar_embedding_prefix = similar_embedding_prefix
        self._normalize_embeddings = normalize_embeddings
        self._output_prefix = output_prefix
        self._tau = tau
        self._loss_function = nn.CrossEntropyLoss(reduction='mean')

    def _compute_partial_loss(self, fst_embeddings, snd_embeddings):
        batch_size = fst_embeddings.shape[0]

        combined_embeddings = torch.cat(
            (fst_embeddings, snd_embeddings),
            dim=0
        )  # (2 * x, embedding_dim)

        if self._normalize_embeddings:
            combined_embeddings = torch.nn.functional.normalize(
                combined_embeddings, p=2, dim=-1, eps=1e-6
            )

        similarity_scores = torch.mm(
            combined_embeddings,
            combined_embeddings.T
        ) / self._tau  # (2 * x, 2 * x)

        positive_samples = torch.cat(
            (torch.diag(similarity_scores, batch_size), torch.diag(similarity_scores, -batch_size)),
            dim=0
        ).reshape(2 * batch_size, 1)  # (2 * x, 1)

        # TODO optimize
        mask = torch.ones(2 * batch_size, 2 * batch_size, dtype=torch.bool)  # (2 * x, 2 * x)
        mask = mask.fill_diagonal_(0)  # Remove equal embeddings scores
        for i in range(batch_size):  # Remove positives
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0

        negative_samples = similarity_scores[mask].reshape(2 * batch_size, -1)  # (2 * x, 2 * x - 2)

        labels = torch.zeros(2 * batch_size).to(positive_samples.device).long()  # (2 * x)
        logits = torch.cat((positive_samples, negative_samples), dim=1)  # (2 * x, 2 * x - 1)

        loss = self._loss_function(logits, labels) / 2  # (1)

        return loss

    def forward(self, inputs):
        original_embeddings = inputs[self._original_embedding_prefix]  # (x, embedding_dim)
        dropout_embeddings = inputs[self._dropout_embedding_prefix]  # (x, embedding_dim)
        similar_embeddings = inputs[self._similar_embedding_prefix]  # (x, embedding_dim)

        dropout_loss = self._compute_partial_loss(original_embeddings, dropout_embeddings)
        ssl_loss = self._compute_partial_loss(original_embeddings, similar_embeddings)

        loss = dropout_loss + ssl_loss

        if self._output_prefix is not None:
            inputs[f'{self._output_prefix}_dropout'] = dropout_loss.cpu().item()
            inputs[f'{self._output_prefix}_ssl'] = ssl_loss.cpu().item()
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class MCLSRLoss(TorchLoss, config_name='mclsr'):

    def __init__(
            self,
            all_scores_prefix,
            mask_prefix,
            normalize_embeddings=False,
            tau=1.0,
            output_prefix=None
    ):
        super().__init__()
        self._all_scores_prefix = all_scores_prefix
        self._mask_prefix = mask_prefix
        self._normalize_embeddings = normalize_embeddings
        self._output_prefix = output_prefix
        self._tau = tau

    def forward(self, inputs):
        all_scores = inputs[self._all_scores_prefix]  # (batch_size, batch_size, seq_len)
        mask = inputs[self._mask_prefix]  # (batch_size)

        batch_size = mask.shape[0]
        seq_len = mask.shape[1]

        positive_mask = torch.eye(batch_size, device=mask.device).bool()

        positive_scores = all_scores[positive_mask]  # (batch_size, seq_len)
        negative_scores = torch.reshape(
            all_scores[~positive_mask],
            shape=(batch_size, batch_size - 1, seq_len)
        )  # (batch_size, batch_size - 1, seq_len)
        assert torch.allclose(all_scores[0, 1], negative_scores[0, 0])
        assert torch.allclose(all_scores[-1, -2], negative_scores[-1, -1])
        assert torch.allclose(all_scores[0, 0], positive_scores[0])
        assert torch.allclose(all_scores[-1, -1], positive_scores[-1])

        # Maybe try mean over sequence TODO
        loss = torch.sum(
            torch.log(torch.sigmoid(positive_scores.unsqueeze(1) - negative_scores))
        )  # (1)

        if self._output_prefix is not None:
            inputs[self._output_prefix] = loss.cpu().item()

        return loss


class IdentityMapLoss(TorchLoss, config_name='identity_map'):

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