import pickle

import numpy as np
import torch
import torch.utils.data as data


def reorder_embeddings_for_letter(embeddings, item_id_to_index):
    max_item_id = max(item_id_to_index.keys())
    num_items = len(item_id_to_index)

    print(f"Max item_id: {max_item_id}")
    print(f"Number of unique items in mapping: {num_items}")

    emb_dim = embeddings.shape[1]

    device = embeddings.device
    new_embeddings = torch.zeros((max_item_id, emb_dim), device=device, dtype=embeddings.dtype)

    for item_id, old_index in item_id_to_index.items():
        new_embeddings[item_id - 1] = embeddings[old_index]

    valid_mask = np.zeros(max_item_id, dtype=bool)
    for item_id in item_id_to_index.keys():
        valid_mask[item_id - 1] = True

    return new_embeddings, valid_mask


class EmbDataset(data.Dataset):

    def __init__(self, dataset_path):
        self.data_path = dataset_path

        with open(dataset_path, 'rb') as f:
            data_reduced = pickle.load(f)

        embeddings_np = data_reduced['embedding']
        original_item_ids = data_reduced['item_id']
        item_id_to_index = {item_id: idx for idx, item_id in enumerate(original_item_ids)}
        text_embeddings = torch.tensor(embeddings_np, dtype=torch.float32)
        self.embeddings = reorder_embeddings_for_letter(text_embeddings, item_id_to_index)[0].detach()
        # print("item_id_to_index", item_id_to_index)
        # self.embeddings = np.fromfile(data_path, dtype=np.float32).reshape(16859,-1)

        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        return self.embeddings[index], index

    def __len__(self):
        return self.embeddings.shape[0]
