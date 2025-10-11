import torch
import torch.nn as nn


class TorchModel(nn.Module):
    @torch.no_grad()
    def _init_weights(self, initializer_range):
        for key, value in self.named_parameters():
            if 'weight' in key:
                if 'norm' in key:
                    nn.init.ones_(value.data)
                else:
                    nn.init.trunc_normal_(
                        value.data,
                        std=initializer_range,
                        a=-2 * initializer_range,
                        b=2 * initializer_range
                    )
            elif 'bias' in key:
                nn.init.zeros_(value.data)
            elif 'codebook' in key:
                nn.init.trunc_normal_(
                    value.data,
                    std=initializer_range,
                    a=-2 * initializer_range,
                    b=2 * initializer_range
                )
            elif "bos_embedding" in key:
                nn.init.trunc_normal_(
                    value.data,
                    std=initializer_range,
                    a=-2 * initializer_range,
                    b=2 * initializer_range,
                )
            else:
                raise ValueError(f'Unknown transformer weight: {key}')