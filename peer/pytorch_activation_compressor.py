from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompressorStats:
    encoded_payloads: int
    decoded_payloads: int


class PyTorchActivationCompressor:
    """Deterministic linear projection compressor for activation tensors."""

    def __init__(
        self,
        *,
        torch_module,
        hidden_size: int,
        latent_dim: int,
        device,
        dtype,
        seed: int = 11,
    ):
        self._torch = torch_module
        self.hidden_size = max(1, int(hidden_size))
        self.latent_dim = max(1, min(int(latent_dim), self.hidden_size))
        self._device = device
        self._dtype = dtype
        self._encoded_payloads = 0
        self._decoded_payloads = 0

        nn = torch_module.nn
        self._encoder = nn.Linear(self.hidden_size, self.latent_dim, bias=False)
        self._decoder = nn.Linear(self.latent_dim, self.hidden_size, bias=False)
        self._initialize_weights(seed=seed)
        self._encoder.to(device=self._device, dtype=self._dtype)
        self._decoder.to(device=self._device, dtype=self._dtype)
        self._encoder.eval()
        self._decoder.eval()

    def _initialize_weights(self, *, seed: int) -> None:
        if self.latent_dim == self.hidden_size:
            eye = self._torch.eye(self.hidden_size, dtype=self._torch.float32)
            self._encoder.weight.data.copy_(eye)
            self._decoder.weight.data.copy_(eye)
            return

        generator = self._torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        basis = self._torch.randn(self.hidden_size, self.latent_dim, generator=generator, dtype=self._torch.float32)
        # Orthonormal basis yields stable projection/reconstruction.
        q, _ = self._torch.linalg.qr(basis, mode="reduced")
        self._encoder.weight.data.copy_(q.transpose(0, 1).contiguous())
        self._decoder.weight.data.copy_(q.contiguous())

    def encode(self, hidden):
        latent = self._encoder(hidden.to(dtype=self._dtype))
        self._encoded_payloads += 1
        return latent

    def decode(self, latent):
        hidden = self._decoder(latent.to(dtype=self._dtype))
        self._decoded_payloads += 1
        return hidden

    def stats(self) -> CompressorStats:
        return CompressorStats(
            encoded_payloads=int(self._encoded_payloads),
            decoded_payloads=int(self._decoded_payloads),
        )
