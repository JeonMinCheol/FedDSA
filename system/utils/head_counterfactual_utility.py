"""Frozen-backbone utility: training MCL gap ON, virtual agreement OFF."""
import copy

import torch
import torch.nn.functional as F


def class_means(values, labels, num_classes):
    sums = values.new_zeros(num_classes)
    counts = values.new_zeros(num_classes)
    sums.index_add_(0, labels, values.detach())
    counts.index_add_(0, labels, torch.ones_like(values))
    return (sums / counts.clamp_min(1)).cpu(), counts.cpu()


def supervision_loss(client, model, features, labels, logits, deployed=False):
    branch = copy.copy(client)
    branch.utility_local_gate = bool(deployed and client.utility_local_gate)
    branch.capture_utility_gate = False
    return branch.anchor_loss(features, labels, logits.detach())[0]


def head_gains(client, model, support, query, validation=None, diagnostic=None):
    modes = {module: module.training for module in model.modules()}
    try:
        model.eval()
        batches = [support, query] + ([validation] if validation is not None else [])
        if diagnostic is not None:
            if validation is None:
                raise ValueError("RAAR B4 requires the separate B3 query")
            batches.append(diagnostic)
        with torch.no_grad():
            features = [model.extract_backbone(x).detach().float() for x, _ in batches]
        if getattr(client, "candidate_gain_enabled", False):
            if validation is None:
                raise ValueError("RAAR requires disjoint B1/B2/B3 training batches")
            next_local_gap = client.nearest_gap_from_dict(client.protos)
            if not torch.isfinite(torch.tensor(next_local_gap)) or next_local_gap < 0:
                raise ValueError("Post-local prototype gap must be finite and nonnegative")
            client.anchor_candidate_evidence = {
                "round": int(client.current_round),
                "support_features": features[0].cpu().clone(),
                "support_labels": support[1].detach().cpu().clone(),
                "allocation_query_features": features[1].cpu().clone(),
                "allocation_query_labels": query[1].detach().cpu().clone(),
                "query_features": features[2].cpu().clone(),
                "query_labels": validation[1].detach().cpu().clone(),
                "mcl_gap": client.mcl_gap_value(), "mcl_gap_class_cap": True,
                "candidate_next_gap": True, "next_local_gap": next_local_gap,
                "next_gap_mode": client.mcl_gap_mode,
                "next_gap_policy": {"server_only": True, "local_utility_unchanged": True,
                                    "local_sgd_gap_unchanged": True},
            }
            if diagnostic is not None:
                client.anchor_candidate_evidence.update(
                    diagnostic_query_features=features[3].cpu().clone(),
                    diagnostic_query_labels=diagnostic[1].detach().cpu().clone())
        projector, head = model.shared_projector, model.shared_head
        params = [projector.weight, projector.bias, head.weight]
        if head.bias is not None:
            params.append(head.bias)

        def classify(z, values):
            if model.shared_classifier_mode == 'linear':
                return F.linear(z, values[2], values[3])
            return model.shared_classifier_scale * (F.normalize(z, dim=1) @ F.normalize(values[2], dim=1).t())

        z = F.linear(features[0], params[0], params[1])
        logits = classify(z, params)
        task = F.cross_entropy(logits, support[1])
        # Fixed utility intervention: MCL gap ON, agreement weighting OFF.
        virtual_client = copy.copy(client)
        virtual_client.agreement_weighting = False
        anchor = supervision_loss(virtual_client, model, z, support[1], logits)
        if not torch.isfinite(task + anchor):
            raise FloatingPointError('Nonfinite frozen-backbone utility support loss')
        with_grads = torch.autograd.grad(task + anchor, params, retain_graph=True)
        without_grads = torch.autograd.grad(task, params)
        with torch.no_grad():
            with_params = [p - client.utility_inner_lr * g for p, g in zip(params, with_grads)]
            without_params = [p - client.utility_inner_lr * g for p, g in zip(params, without_grads)]
            result = []
            for feat, (_, labels) in zip(features[1:], batches[1:]):
                def losses(values):
                    output = classify(F.linear(feat, values[0], values[1]), values)
                    return F.cross_entropy(output, labels, reduction='none')
                gain = losses(without_params) - losses(with_params)
                if not torch.isfinite(gain).all():
                    raise FloatingPointError('Nonfinite frozen-backbone utility query gain')
                result.append(class_means(gain, labels, client.num_classes))
        return result
    finally:
        for module, training in modes.items():
            module.training = training
