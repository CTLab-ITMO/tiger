import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleVectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta=0.25):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        z: [batch_size, hidden_dim]
        return: quantized z, loss, indices
        """
        distances = torch.sum(z ** 2, dim=1, keepdim=True) + \
                    torch.sum(self.embedding.weight ** 2, dim=1) - \
                    2 * torch.matmul(z, self.embedding.weight.t())

        indices = torch.argmin(distances, dim=-1)
        z_q = self.embedding(indices)

        # VQ-VAE loss
        commitment_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss = F.mse_loss(z_q, z.detach())
        loss = codebook_loss + self.beta * commitment_loss

        z_q = z + (z_q - z).detach()

        return z_q, loss, indices


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, num_emb_list, e_dim, beta=0.25):
        super().__init__()
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.num_quantizers = len(num_emb_list)

        self.vq_layers = nn.ModuleList([
            SimpleVectorQuantizer(n_e, e_dim, beta=beta)
            for n_e in num_emb_list
        ])

    def forward(self, z):
        """
        z: [batch_size, hidden_dim]
        return: quantized z, total loss, all indices
        """
        all_losses = []
        all_indices = []

        z_q = 0
        residual = z

        # Итеративно квантуем residual на каждом уровне
        for quantizer in self.vq_layers:
            z_res, loss, indices = quantizer(residual)
            residual = residual - z_res
            z_q = z_q + z_res

            all_losses.append(loss)
            all_indices.append(indices)

        total_loss = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)  # [batch_size, num_quantizers]

        return z_q, total_loss, all_indices


class SimpleMLP(nn.Module):
    def __init__(self, layers, dropout=0.0):
        super().__init__()
        self.layers = layers

        mlp_modules = []
        for i, (in_dim, out_dim) in enumerate(zip(layers[:-1], layers[1:])):
            mlp_modules.append(nn.Linear(in_dim, out_dim))
            # ReLU для всех слоев кроме последнего
            if i < len(layers) - 2:
                mlp_modules.append(nn.ReLU())
                if dropout > 0:
                    mlp_modules.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*mlp_modules)

        # Xavier init
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.0)

    def forward(self, x):
        return self.mlp(x)


class SimpleRQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 num_emb_list=[256, 256, 256],
                 e_dim=32,
                 hidden_layers=[512, 256, 128],
                 beta=0.25,
                 loss_type="mse"):
        super().__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.beta = beta
        self.loss_type = loss_type

        # Encoder: input_dim -> hidden_layers -> e_dim
        encoder_layers = [in_dim] + hidden_layers + [e_dim]
        self.encoder = SimpleMLP(encoder_layers)

        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim, beta=beta)

        # Decoder: e_dim -> hidden_layers (reversed) -> input_dim
        decoder_layers = [e_dim] + hidden_layers[::-1] + [in_dim]
        self.decoder = SimpleMLP(decoder_layers)

    def forward(self, x):
        """
        x: [batch_size, in_dim] - input embeddings
        returns: reconstructed x, total loss, semantic indices, quantized representation
        """
        z = self.encoder(x)
        z_q, rq_loss, indices = self.rq(z)
        x_recon = self.decoder(z_q)

        return x_recon, rq_loss, indices, z_q

    def get_semantic_ids(self, x):
        with torch.no_grad():
            z = self.encoder(x)
            _, _, indices = self.rq(z)
            return indices

    def compute_loss(self, x_recon, x, rq_loss):
        """
        Compute total loss (reconstruction + quantization)
        """
        if self.loss_type == "mse":
            recon_loss = F.mse_loss(x_recon, x)
        elif self.loss_type == "l1":
            recon_loss = F.l1_loss(x_recon, x)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        total_loss = recon_loss + rq_loss
        return total_loss, recon_loss

    def resolve_collisions(self, indices):
        """
        доп токен как в TIGER: "To avoid multiple items being mapped to the same
        Semantic ID, we add a unique 4th code for items that share the same first
        three codewords"
        """
        batch_size = indices.size(0)
        device = indices.device

        # Конвертируем indices в строки для поиска дубликатов
        indices_str = []
        for i in range(batch_size):
            id_str = "-".join([str(idx.item()) for idx in indices[i]])
            indices_str.append(id_str)

        # Создаем уникальные ID добавляя четвертый токен
        unique_indices = []
        id_counts = {}

        for i, id_str in enumerate(indices_str):
            if id_str not in id_counts:
                id_counts[id_str] = 0
                collision_token = 0
            else:
                id_counts[id_str] += 1
                collision_token = id_counts[id_str]

            # Добавляем четвертый токен для разрешения коллизий
            unique_id = torch.cat([indices[i], torch.tensor([collision_token],
                                                            dtype=indices.dtype,
                                                            device=device)])
            unique_indices.append(unique_id)

        return torch.stack(unique_indices)
