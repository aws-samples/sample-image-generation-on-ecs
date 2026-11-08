import torch.nn as nn


class DecoderWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        z = self.vae.post_quant_conv(z) if self.vae.post_quant_conv is not None else z
        return self.vae.decoder(z)
