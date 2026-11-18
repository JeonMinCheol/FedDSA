"""Federated variation generator: a tiny class-conditional VAE on images, identical in every client (FedAvg'd
across heterogeneous backbones). It is trained only to model the data (ELBO), never against the classifier, so it
cannot collapse to a universal adversarial shortcut. Augmentation keeps the real image sharp and adds the learned
within-class variation: x' = x + s * (dec(z_sample, y) - dec(mu, y))."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VarGen(nn.Module):
    def __init__(self, num_classes, ch=3, zc=16, width=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(ch, width // 2, 4, 2, 1), nn.ReLU(),
                                 nn.Conv2d(width // 2, width, 4, 2, 1), nn.ReLU(),
                                 nn.Conv2d(width, 2 * zc, 3, 1, 1))
        self.dec_in = nn.Conv2d(zc, width, 3, 1, 1)
        self.emb = nn.Embedding(num_classes, width)  # class enters the decoder: latent keeps within-class variation
        self.dec = nn.Sequential(nn.ReLU(), nn.ConvTranspose2d(width, width // 2, 4, 2, 1), nn.ReLU(),
                                 nn.ConvTranspose2d(width // 2, ch, 4, 2, 1))

    def decode(self, z, y):
        return self.dec(self.dec_in(z) + self.emb(y)[:, :, None, None])

    def elbo_loss(self, x, y):
        mu, logvar = self.enc(x).chunk(2, 1)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        rec = F.mse_loss(self.decode(z, y), x, reduction="none").flatten(1).sum(1)
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).flatten(1).sum(1)
        return (rec + kl).mean() / x[0].numel(), float(rec.mean() / x[0].numel())

    @torch.no_grad()
    def encode_mu(self, x):
        return self.enc(x).chunk(2, 1)[0]

    @torch.no_grad()
    def vary_toward(self, x, y, mean, std, mix=0.5, scale=1.0):
        """Move my image's latent part of the way toward a sample of the class's latent distribution (mean/std:
        [C, zc, h, w], pooled over clients when the generator is shared), and add the decoded difference to the sharp
        real image: x' = x + s * (dec(mu_x + lam * (z_c - mu_x), y) - dec(mu_x, y)), lam ~ U(0, mix)."""
        mu = self.enc(x).chunk(2, 1)[0]
        zc = mean[y] + std[y] * torch.randn_like(mu)
        lam = mix * torch.rand(x.shape[0], 1, 1, 1, device=x.device)
        return x + scale * (self.decode(mu + lam * (zc - mu), y) - self.decode(mu, y))

    @torch.no_grad()
    def transport(self, x, y, gm, gs, lm, ls, mix=1.0, scale=1.0):
        """Latent moment transport (AdaIN in the generator's latent): keep my image's position inside MY class
        distribution but move that distribution onto the federation's: z* = gm_y + (mu_x - lm_y) * gs_y / ls_y,
        z = mu_x + lam * (z* - mu_x), lam ~ U(0, mix); x' = x + s * (dec(z) - dec(mu_x)). Only the difference between
        the pooled and my class statistics enters, not sampling noise (which swamped it in vary_toward)."""
        mu = self.enc(x).chunk(2, 1)[0]
        ratio = (gs[y] / ls[y].clamp(min=1e-3)).clamp(0.5, 2.0)  # classes with few local samples have tiny std
        zs = gm[y] + (mu - lm[y]) * ratio
        lam = mix * torch.rand(x.shape[0], 1, 1, 1, device=x.device)
        return x + scale * (self.decode(mu + lam * (zs - mu), y) - self.decode(mu, y))

    @staticmethod
    def lowfreq_noise(x, mag=0.1, res=8):
        """Control: smooth random perturbation (noise at res x res, bilinearly upsampled) with mean |delta| = mag."""
        n = torch.randn(x.shape[0], x.shape[1], res, res, device=x.device)
        n = F.interpolate(n, size=x.shape[2:], mode="bilinear", align_corners=False)
        return x + mag * n / n.abs().mean((1, 2, 3), keepdim=True)

    @torch.no_grad()
    def vary(self, x, y, scale=1.0, temp=1.0):
        mu, logvar = self.enc(x).chunk(2, 1)
        z = mu + temp * torch.randn_like(mu) * (0.5 * logvar).exp()
        return x + scale * (self.decode(z, y) - self.decode(mu, y))


if __name__ == "__main__":  # self-check: ELBO decreases on a toy batch; vary() keeps shape
    torch.manual_seed(0)
    g = VarGen(10)
    x, y = torch.randn(32, 3, 32, 32), torch.randint(10, (32,))
    opt = torch.optim.Adam(g.parameters(), 1e-3)
    l0 = g.elbo_loss(x, y)[0].item()
    for _ in range(50):
        l, _ = g.elbo_loss(x, y)
        opt.zero_grad(); l.backward(); opt.step()
    assert l.item() < l0 and g.vary(x, y).shape == x.shape
    print(f"ok elbo {l0:.3f} -> {l.item():.3f}")
