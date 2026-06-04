"""VAE wrappers adapting ``AutoencoderKL`` to a flat Core ML tensor contract.

Two thin modules, one per direction, mirroring ``CoreMLUNetWrapper``: they expose
a single positional tensor in / single tensor out so the traced graph has a clean,
named Core ML I/O signature. They call the VAE *submodules* directly
(``post_quant_conv``/``decoder``, ``encoder``/``quant_conv``) rather than
``decode``/``encode`` — the same op graph, but free of the ``return_dict``
plumbing and the ``DiagonalGaussianDistribution`` wrapper that complicate tracing.

Scaling is intentionally NOT baked in. The pipeline owns ``scaling_factor``
(``latent = latent / scaling_factor`` before decode, ``moments`` -> distribution
-> sample -> ``* scaling_factor`` after encode), keeping these artifacts 1:1 with
the reference VAE.

The mid-block self-attention is converted via the ORIGINAL (full, fp32-score)
processor; see ``convert.convert_vae_*``.
"""

import torch


class CoreMLVAEDecoderWrapper(torch.nn.Module):
    """latent ``(B, latent_channels, h, w)`` -> image ``(B, 3, h*8, w*8)``."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent):
        z = self.vae.post_quant_conv(latent)
        return self.vae.decoder(z)


class CoreMLVAEEncoderWrapper(torch.nn.Module):
    """image ``(B, 3, h*8, w*8)`` -> latent moments ``(B, 2*latent_channels, h, w)``.

    Outputs the raw moments (mean ‖ logvar) exactly as ``AutoencoderKL.encode``
    produces them before wrapping in a ``DiagonalGaussianDistribution``. Sampling
    (mean + std·noise) is deferred to the pipeline so the converted encoder stays
    deterministic and noise-source agnostic.
    """

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, image):
        h = self.vae.encoder(image)
        return self.vae.quant_conv(h)
