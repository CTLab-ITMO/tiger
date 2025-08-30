import torch


class BaseMetric:
    pass


class StatefullMetric(BaseMetric):
    def reduce(self, values):
        raise NotImplementedError


class StaticMetric(BaseMetric):
    def __init__(self, name, value):
        self._name = name
        self._value = value

    def __call__(self, inputs):
        inputs[self._name] = self._value

        return inputs


class CompositeMetric(BaseMetric):

    def __init__(self, metrics):
        self._metrics = metrics

    def __call__(self, inputs):
        for metric in self._metrics:
            inputs = metric(inputs)
        return inputs


class NDCGMetric(BaseMetric):

    def __init__(self, k):
        self._k = k

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix][:, :self._k].float()  # (batch_size, top_k_indices)
        labels = inputs['{}.ids'.format(labels_prefix)].float()  # (batch_size)

        assert labels.shape[0] == predictions.shape[0]

        hits = torch.eq(predictions, labels[..., None]).float()  # (batch_size, top_k_indices)
        discount_factor = 1 / torch.log2(torch.arange(1, self._k + 1, 1).float() + 1.).to(hits.device)  # (k)
        dcg = torch.einsum('bk,k->b', hits, discount_factor)  # (batch_size)

        return dcg.cpu().tolist()


class NDCGSemanticMetric(BaseMetric):

    def __init__(self, k):
        self._k = k

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix].long()

        batch_size, _, sid_length = predictions.shape

        labels = inputs['semantic_{}.ids'.format(labels_prefix)].long()  # (batch_size)
        labels = labels.reshape(batch_size, 1, sid_length)
        offsetted_labels = labels + 256 * torch.arange(4, device=labels.device)[None, None, :]

        hits = (torch.eq(predictions[:, :self._k, :], offsetted_labels).sum(
            dim=-1) == sid_length).float()  # (batch_size, top_k_indices)

        discount_factor = 1 / torch.log2(torch.arange(1, self._k + 1, 1).float() + 1.).to(hits.device)  # (k)
        dcg = torch.einsum('bk,k->b', hits, discount_factor)  # (batch_size)

        return dcg.cpu().tolist()


class RecallMetric(BaseMetric):

    def __init__(self, k):
        self._k = k

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix][:, :self._k].float()  # (batch_size, top_k_indices)
        labels = inputs['{}.ids'.format(labels_prefix)].float()  # (batch_size)

        assert labels.shape[0] == predictions.shape[0]

        hits = torch.eq(predictions, labels[..., None]).float()  # (batch_size, top_k_indices)
        recall = hits.sum(dim=-1)  # (batch_size)

        return recall.cpu().tolist()


class RecallSemanticMetric(BaseMetric):

    def __init__(self, k):
        self._k = k

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix].long()

        batch_size, _, sid_length = predictions.shape

        labels = inputs['semantic_{}.ids'.format(labels_prefix)].long()  # (batch_size)
        labels = labels.reshape(batch_size, 1, sid_length)
        offsetted_labels = labels + 256 * torch.arange(4, device=labels.device)[None, None, :]

        hits = (torch.eq(predictions[:, :self._k, :], offsetted_labels).sum(
            dim=-1) == sid_length).float()  # (batch_size, top_k_indices)
        recall = hits.sum(dim=-1)  # (batch_size)

        return recall.cpu().tolist()


class CoverageMetric(StatefullMetric):

    def __init__(self, k, num_items):
        self._k = k
        self._num_items = num_items

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix][:, :self._k].float()  # (batch_size, top_k_indices)
        return predictions.view(-1).long().cpu().detach().tolist()  # (batch_size * k)

    def reduce(self, values):
        return len(set(values)) / self._num_items


class CoverageSemanticMetric(StatefullMetric):

    def __init__(self, k, num_items):
        self._k = k
        self._num_items = num_items

    def __call__(self, inputs, pred_prefix, labels_prefix):
        predictions = inputs[pred_prefix].long()
        predictions = predictions[:, :self._k, :].sum(dim=-1).long()
        return predictions.view(-1).long().cpu().detach().tolist()  # (batch_size * k)

    def reduce(self, values):
        return len(set(values)) / self._num_items
