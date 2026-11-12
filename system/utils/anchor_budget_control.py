"""Counterfactual weight allocation at an externally fixed per-round L2 budget."""
import json
import math
from pathlib import Path
import time

import torch


class ReferenceBudgetStream:
    """Read actual reference displacements; incomplete writes never count as a round."""
    def __init__(self, directory, seed, gamma, row_validator=None):
        self.directory = Path(directory)
        self.seed = int(seed)
        self.gamma = float(gamma)
        self.offset = 0
        self.budgets = {}
        self.row_validator = row_validator

    def refresh(self):
        path = self.directory/'utility_gates.jsonl'
        if not path.exists():
            return
        with path.open('rb') as handle:
            handle.seek(self.offset)
            while True:
                start = handle.tell()
                line = handle.readline()
                if not line or not line.endswith(b'\n'):
                    self.offset = start
                    break
                row = json.loads(line)
                if row['seed'] != self.seed or abs(row['anchor_candidate_mix']-self.gamma)>1e-12:
                    raise ValueError('Reference seed or prototype candidate gamma mismatch')
                values = row['anchor_step_norm']
                if not values or any(v is None or not math.isfinite(v) or v<0 for v in values):
                    raise ValueError('Reference movement budget must be finite and nonnegative')
                if self.row_validator is not None:
                    self.row_validator(row)
                round_idx = int(row['round'])
                if round_idx in self.budgets:
                    raise ValueError('Duplicate reference budget round')
                self.budgets[round_idx] = {'mean_l2': sum(values)/len(values), 'num_classes':len(values),
                    'class_l2':values,'class_index':(list(range(len(values)))
                        if row.get('class_index')=='arrays are indexed by class label; null means unavailable'
                        else row.get('class_index')),
                    'candidate_max_weight':row.get('candidate_update_trust',{}).get('max_weight')}
                self.offset = handle.tell()

    def get(self, round_idx, wait=True):
        last_notice = 0
        while True:
            self.refresh()
            if round_idx in self.budgets:
                return self.budgets[round_idx]
            if not wait:
                return None
            status = json.loads((self.directory/'status.json').read_text())
            if status.get('status') in {'failed','stopped','complete'}:
                raise RuntimeError(f'Reference ended without movement budget for round {round_idx}')
            pid = status.get('pid')
            if pid:
                try:
                    state = Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]
                except FileNotFoundError:
                    state = None
                if state is None or state=='Z':
                    # Re-read after observing termination in case the final record just arrived.
                    self.refresh()
                    if round_idx in self.budgets:
                        return self.budgets[round_idx]
                    raise RuntimeError(f'Reference process ended before budget round {round_idx}')
            if time.monotonic()-last_notice>60:
                print(f'Waiting for live reference movement budget: round={round_idx}',flush=True)
                last_notice=time.monotonic()
            time.sleep(5)


@torch.no_grad()
def budget_matched_anchor_update(current, candidate, proposed_weights, observed,
                                 target_mean_l2, mode, permutation_seed):
    """Match sum_c ||next_c-current_c|| to the recorded reference exactly in real arithmetic.

    Uniform weights remove class-specific reliability. Shuffled weights break
    score-to-class assignment with a private RNG. Global rescaling matches the
    physical displacement budget, not the arithmetic mean of the coefficients.
    """
    if mode not in {'budget_uniform','budget_shuffled','budget_aligned'}:
        raise ValueError('Unknown displacement allocation control')
    if not math.isfinite(target_mean_l2) or target_mean_l2<0:
        raise ValueError('Target displacement budget must be finite and nonnegative')
    if current.ndim!=2 or candidate.shape!=current.shape:
        raise ValueError('Current and candidate anchor shapes must match')
    if current.dtype!=candidate.dtype or current.device!=candidate.device:
        raise ValueError('Anchor dtypes and devices must match')
    if proposed_weights.shape!=current.shape[:1] or observed.shape!=proposed_weights.shape:
        raise ValueError('Weights and observation mask must contain one entry per class')
    if not torch.isfinite(current).all() or not torch.isfinite(candidate).all():
        raise ValueError('Anchors must be finite')
    mask=observed.to(device=current.device,dtype=torch.bool)
    weights=proposed_weights.to(device=current.device,dtype=current.dtype)
    if not torch.isfinite(weights[mask]).all() or (weights[mask]<0).any():
        raise ValueError('Proposed observed weights must be finite and nonnegative')
    indices=torch.nonzero(mask,as_tuple=False).flatten()
    sources=indices.clone()
    coefficients=torch.zeros_like(weights)
    if mode=='budget_uniform':
        coefficients[mask]=1
    elif mode=='budget_aligned':
        coefficients[mask]=weights[mask]
    elif indices.numel():
        generator=torch.Generator(device='cpu')
        generator.manual_seed(int(permutation_seed)%(2**63-1))
        permutation=torch.randperm(indices.numel(),generator=generator).to(indices.device)
        sources=indices.index_select(0,permutation)
        coefficients[indices]=weights.index_select(0,sources)
    delta=candidate-current
    proposed_total=(coefficients.double()*delta.double().norm(dim=1)).sum()
    target_total=float(target_mean_l2)*current.size(0)
    if target_total==0:
        scale=0.0
    elif proposed_total.item()<=0:
        raise ValueError('Positive reference budget cannot be assigned to zero candidate movement')
    else:
        scale=target_total/proposed_total.item()
    coefficients*=scale
    updated=current+coefficients[:,None]*delta
    realized=(updated-current).norm(dim=1).mean().item()
    tolerance=max(1e-7,abs(target_mean_l2)*1e-5)
    if abs(realized-target_mean_l2)>tolerance:
        raise RuntimeError('Float precision prevented matching the actual displacement budget')
    applied=coefficients[mask]
    summary={
        'mode':mode,'application_reference':'current_round_anchor','mean_normalization':False,
        'allocation_control':True,'budget_source':'external adaptive reference actual per-round mean L2',
        'target_mean_l2':float(target_mean_l2),'realized_mean_l2':float(realized),
        'budget_abs_error':abs(realized-target_mean_l2),'budget_scale':float(scale),
        'observed_classes':int(indices.numel()),
        'observed_coefficient_mean':float(applied.mean()) if applied.numel() else None,
        'observed_coefficient_min':float(applied.min()) if applied.numel() else None,
        'observed_coefficient_max':float(applied.max()) if applied.numel() else None,
        'observed_above_one':int((applied>1).sum()),
        'candidate_step_mean_norm':float(delta.norm(dim=1).mean()),
        'applied_step_mean_norm':float(realized),
        'permutation_seed':int(permutation_seed) if mode=='budget_shuffled' else None,
        'source_class_indices':sources.detach().cpu().tolist() if mode=='budget_shuffled' else None,
        'coefficient_one_is_fixed_ema':True,
    }
    return updated,coefficients,summary
