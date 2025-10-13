import torch


class NDCGMetric:
    def __init__(self, k):
        self._k = k

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix][:, :self._k].float()  # (batch_size, top_k_indices)
        labels = inputs[f'{labels_prefix}.ids'].float()  # (batch_size)

        assert labels.shape[0] == predictions.shape[0]

        hits = torch.eq(predictions, labels[..., None]).float()  # (batch_size, top_k_indices)
        discount_factor = 1 / torch.log2(torch.arange(1, self._k + 1, 1).float() + 1.).to(hits.device)  # (k)
        dcg = torch.einsum('bk,k->b', hits, discount_factor)  # (batch_size)

        return dcg.cpu().tolist()


class NDCGSemanticMetric:
    def __init__(self, k, codebook_size, num_codebooks):
        self._k = k
        self._codebook_size = codebook_size
        self._num_codebooks = num_codebooks

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix].long()

        batch_size, _, sid_length = predictions.shape

        labels = inputs[f'semantic_{labels_prefix}.ids'].long()  # (batch_size)
        labels = labels.reshape(batch_size, 1, sid_length)
        offsetted_labels = labels + self._codebook_size * torch.arange(self._num_codebooks, device=labels.device)[None, None, :]

        hits = (torch.eq(predictions[:, :self._k, :], offsetted_labels).sum(dim=-1) == sid_length).float() # (batch_size, top_k_indices)

        discount_factor = 1 / torch.log2(torch.arange(1, self._k + 1, 1).float() + 1.).to(hits.device)  # (k)
        dcg = torch.einsum('bk,k->b', hits, discount_factor)  # (batch_size)

        return dcg.cpu().tolist()


class RecallMetric:
    def __init__(self, k):
        self._k = k

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix][:, :self._k].float()  # (batch_size, top_k_indices)
        labels = inputs[f'{labels_prefix}.ids'].float()  # (batch_size)

        assert labels.shape[0] == predictions.shape[0]

        hits = torch.eq(predictions, labels[..., None]).float()  # (batch_size, top_k_indices)
        recall = hits.sum(dim=-1)  # (batch_size)

        return recall.cpu().tolist()


class RecallSemanticMetric:
    def __init__(self, k, codebook_size, num_codebooks):
        self._k = k
        self._codebook_size = codebook_size
        self._num_codebooks = num_codebooks

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix].long()

        batch_size, _, sid_length = predictions.shape

        labels = inputs[f'semantic_{labels_prefix}.ids'].long()  # (batch_size)
        labels = labels.reshape(batch_size, 1, sid_length)
        offsetted_labels = labels + self._codebook_size * torch.arange(self._num_codebooks, device=labels.device)[None, None, :]

        hits = (torch.eq(predictions[:, :self._k, :], offsetted_labels).sum(dim=-1) == sid_length).float() # (batch_size, top_k_indices)
        recall = hits.sum(dim=-1)  # (batch_size)

        return recall.cpu().tolist()
