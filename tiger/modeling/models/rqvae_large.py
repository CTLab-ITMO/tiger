import functools

import faiss
import torch

from models.base import TorchModel
from utils import DEVICE


class RqVaeModelLarge(TorchModel, config_name="rqvae_large"):
    def __init__(
        self,
        train_sampler,
        input_dim: int,
        intermediate_dims: list[int],
        hidden_dim: int,
        n_iter: int,
        codebook_sizes: list[int],
        should_init_codebooks,
        should_reinit_unused_clusters,
        initializer_range,
    ):
        super().__init__()

        self.n_iter = n_iter

        # Kmeans initialization
        self.should_init_codebooks = should_init_codebooks

        # Trick with re-initing empty clusters
        self.should_reinit_unused_clusters = should_reinit_unused_clusters

        # Enc and dec are mirrored copies of each other
        self.encoder = self.make_encoding_tower(input_dim, intermediate_dims, hidden_dim)
        self.decoder = self.make_encoding_tower(hidden_dim, intermediate_dims[::-1], input_dim)

        # Default initialization of codebook
        self.codebooks = torch.nn.ParameterList()

        self.codebook_sizes = codebook_sizes

        for codebook_size in codebook_sizes:
            cb = torch.FloatTensor(codebook_size, hidden_dim)
            self.codebooks.append(cb)

        self._init_weights(initializer_range)

        if self.should_init_codebooks:
            if train_sampler is None:
                raise AttributeError("Train sampler is None")

            embeddings = torch.stack(
                [entry["item.embed"] for entry in train_sampler._dataset]
            )
            self.init_codebooks(embeddings)
            print("Codebooks initialized with Faiss Kmeans")
            self.should_init_codebooks = False

    @classmethod
    def create_from_config(cls, config, **kwargs):
        return cls(
            train_sampler=kwargs.get("train_sampler"),
            input_dim=config["embedding_dim"],
            intermediate_dims=config["intermediate_dims"],
            hidden_dim=config["hidden_dim"],
            n_iter=config["n_iter"],
            codebook_sizes=config["codebook_sizes"],
            should_init_codebooks=config.get("should_init_codebooks", False),
            should_reinit_unused_clusters=config.get(
                "should_reinit_unused_clusters", False
            ),
            initializer_range=config.get("initializer_range", 0.02),
        )

    def make_encoding_tower(self, d1: int, intermediate_dims: list[int], d2: int):
        layers = [torch.nn.Linear(d1, intermediate_dims[0], bias=False)]
        for i in range(1, len(intermediate_dims)):
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(intermediate_dims[i - 1], intermediate_dims[i], bias=False))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(intermediate_dims[-1], d2, bias=False))
        return torch.nn.Sequential(*layers)

    @staticmethod
    def get_codebook_indices(remainder, codebook):
        dist = torch.cdist(remainder, codebook)
        return dist.argmin(dim=-1)

    def init_codebooks(self, embeddings):
        with torch.no_grad():
            remainder = self.encoder(embeddings)
            for codebook in self.codebooks:
                embeddings_np = remainder.cpu().numpy()
                n_clusters = codebook.shape[0]

                kmeans = faiss.Kmeans(
                    d=embeddings_np.shape[1],
                    k=n_clusters,
                    niter=self.n_iter,
                    gpu=1,
                )
                kmeans.train(embeddings_np)

                codebook.data = torch.from_numpy(kmeans.centroids).to(codebook.device)

                codebook_indices = self.get_codebook_indices(remainder, codebook)
                codebook_vectors = codebook[codebook_indices]
                remainder = remainder - codebook_vectors

    @staticmethod
    def reinit_unused_clusters(remainder, codebook, codebook_indices):
        with torch.no_grad():
            is_used = torch.full((codebook.shape[0],), False, device=codebook.device)
            unique_indices = codebook_indices.unique()
            is_used[unique_indices] = True
            rand_input = torch.randint(0, remainder.shape[0], ((~is_used).sum(),))
            codebook[~is_used] = remainder[rand_input]

    def train_pass(self, embeddings):
        latent_vector = self.encoder(embeddings)

        latent_restored = 0

        num_unique_clusters = []
        remainder = latent_vector

        remainders = []
        codebooks_vectors = []

        for codebook in self.codebooks:
            remainders.append(remainder)

            codebook_indices = self.get_codebook_indices(remainder, codebook)
            codebook_vectors = codebook[codebook_indices]

            if self.should_reinit_unused_clusters:
                self.reinit_unused_clusters(remainder, codebook, codebook_indices)

            num_unique_clusters.append(codebook_indices.unique().shape[0])

            codebooks_vectors.append(codebook_vectors)

            latent_restored = latent_restored + codebook_vectors
            remainder = remainder - codebook_vectors

        # Here we cast recon loss to latent vector
        latent_restored = latent_vector + (latent_restored - latent_vector).detach()
        embeddings_restored = self.decoder(latent_restored)

        return {
            "embeddings": embeddings,
            "embeddings_restored": embeddings_restored,
            "remainders": remainders,
            "codebooks_vectors": codebooks_vectors,
        }

    def eval_pass(self, embeddings):
        ind_lists = []
        remainder = self.encoder(embeddings)
        for codebook in self.codebooks:
            codebook_indices = self.get_codebook_indices(remainder, codebook)
            codebook_vectors = codebook[codebook_indices]
            ind_lists.append(codebook_indices.cpu().numpy())
            remainder = remainder - codebook_vectors
        return torch.tensor(list(zip(*ind_lists))).to(DEVICE), remainder

    def forward(self, inputs):
        embeddings = inputs["embeddings"]

        if self.training:  # training mode
            return self.train_pass(embeddings)
        else:  # eval mode
            return self.eval_pass(embeddings)

    @functools.cache
    def get_single_embedding(self, codebook_idx: int, codebook_id: int):
        return self.codebooks[codebook_idx][codebook_id]
