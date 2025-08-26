import json

import numpy as np
import torch

from models import RqVaeModel, RqVaeModelLarge
from utils import DEVICE


def test(a, b):
    cos_sim = torch.nn.functional.cosine_similarity(a, b, dim=0)
    norm_a = torch.norm(a, p=2)
    norm_b = torch.norm(b, p=2)
    l2_dist = torch.norm(a - b, p=2) / (norm_a + norm_b + 1e-8)
    return cos_sim, l2_dist


if __name__ == "__main__":
    config = json.load(open("../configs/train/tiger_train_config.json"))
    config = config["model"]
    rqvae_config = json.load(open(config["rqvae_train_config_path"]))
    rqvae_config["model"]["should_init_codebooks"] = False
    rqvae_model = RqVaeModelLarge.create_from_config(rqvae_config["model"]).to(DEVICE)
    rqvae_model.load_state_dict(
        torch.load(config["rqvae_checkpoint_path"], weights_only=True)
    )
    df = torch.load(config["embs_extractor_path"], weights_only=False)
    embeddings_array = np.stack(df["embeddings"].values)
    tensor_embeddings = torch.tensor(
        embeddings_array, dtype=torch.float32, device=DEVICE
    )
    inputs = {"embeddings": tensor_embeddings}

    rqvae_model.eval()
    sem_ids, residuals = rqvae_model.forward(inputs)
    scores = residuals.detach()
    print(torch.norm(residuals, p=2, dim=1).median())
    for i, codebook in enumerate(rqvae_model.codebooks):
        scores += codebook[sem_ids[:, i]].detach()
    decoder_output = rqvae_model.decoder(scores.detach()).detach()

    a = tensor_embeddings[0]
    b = decoder_output[0]
    cos_sim, l2_dist = test(a, b)
    print("косинусное расстояние", cos_sim)
    print("евклидово расстояние", l2_dist)

    cos_sim = torch.nn.functional.cosine_similarity(
        tensor_embeddings, decoder_output, dim=1
    )
    print("косинусное расстояние", cos_sim.mean(), cos_sim.min(), cos_sim.max())

    norm_a = torch.norm(tensor_embeddings, p=2, dim=1)
    norm_b = torch.norm(decoder_output, p=2, dim=1)
    l2_dist = torch.norm(decoder_output - tensor_embeddings, p=2, dim=1) / (
        norm_a + norm_b + 1e-8
    )
    print("евклидово расстояние", l2_dist.median(), l2_dist.min(), l2_dist.max())
