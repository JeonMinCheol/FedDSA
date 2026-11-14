"""Exact bank-only geometry shared by frozen B2/B4 client evaluations."""
import torch
import torch.nn.functional as F

from utils.mcl_gap import _safe_norm, _validate
from utils.raar_next_gap import _constant_gap


class FrozenAnchorGeometry:
    """One immutable, device-local candidate bank; never used for B3 gradients."""

    @torch.no_grad()
    def __init__(self, bank):
        if bank.requires_grad:
            raise ValueError('Frozen geometry cannot replace differentiable anchor geometry')
        _validate(bank, 0.)
        self.bank = bank.detach().clone()
        self.double_bank = self.bank.double()
        if bank.size(0) == 1:
            self.nearest = self.double_bank.sum(dim=1)*0
        else:
            distances = torch.cdist(self.double_bank, self.double_bank, p=2,
                compute_mode='donot_use_mm_for_euclid_dist')
            diagonal = torch.eye(bank.size(0), dtype=torch.bool, device=bank.device)
            self.nearest = distances.masked_fill(diagonal, float('inf')).min(dim=1).values
        magnitude = _safe_norm(self.double_bank)
        if not bool(torch.isfinite(self.nearest).all()) or not bool(torch.isfinite(magnitude).all()):
            raise ValueError('Anchor-distance computation must remain finite')
        scale = torch.maximum(torch.ones_like(self.nearest), torch.maximum(magnitude, self.nearest))
        self.slack = 16*torch.finfo(bank.dtype).eps*scale
        self.global_gap = self.nearest.mean()

    def next_gap(self, old_gap, context):
        if context is None:
            return float(old_gap)
        if not isinstance(context, dict) or 'next_local_gap' not in context or 'mode' not in context:
            raise ValueError('Next-gap context requires next_local_gap and mode')
        local = _constant_gap(context['next_local_gap'], 'next_local_gap')
        mode = context['mode']
        if mode not in ('min', 'max'):
            raise ValueError('next-gap mode must be min or max')
        _validate(self.bank, old_gap)
        if self.bank.size(0) == 1 or local <= 0:
            return (self.double_bank.sum()*0).to(self.bank)
        if bool(self.global_gap.detach() <= 0):
            return (self.global_gap*0).to(self.bank)
        local_gap = self.global_gap.new_tensor(local)
        result = (torch.minimum(local_gap, self.global_gap) if mode == 'min'
                  else torch.maximum(local_gap, self.global_gap))
        return result.to(self.bank)

    def class_gaps(self, base_gap, class_cap):
        _validate(self.bank, base_gap)
        if not class_cap:
            return (base_gap.to(self.bank).expand(self.bank.size(0)) if torch.is_tensor(base_gap)
                    else self.bank.new_full((self.bank.size(0),), float(base_gap)))
        raw = (base_gap.to(self.nearest).expand_as(self.nearest) if torch.is_tensor(base_gap)
               else torch.full_like(self.nearest, float(base_gap)))
        return torch.minimum(raw, (self.nearest-self.slack).clamp_min(0)).to(self.bank)

    def add_gap(self, distances, labels, base_gap, class_cap):
        _validate(self.bank, base_gap)
        if (distances.ndim != 2 or distances.size(1) != self.bank.size(0)
                or labels.ndim != 1 or labels.numel() != distances.size(0)):
            raise ValueError('One true label per distance row and matching classes required')
        if not torch.is_tensor(base_gap) and float(base_gap) <= 0:
            return distances
        one_hot = F.one_hot(labels, self.bank.size(0)).to(device=distances.device, dtype=distances.dtype)
        if not class_cap:
            return distances+one_hot*(base_gap.to(distances) if torch.is_tensor(base_gap) else float(base_gap))
        gaps = self.class_gaps(base_gap, True).to(distances)
        return distances+one_hot*gaps.index_select(0, labels)[:, None]
