"""Infer image geometry once per dataset, before creating trainable parameters."""

import ast

import numpy as np
import torch
from torch import nn

from utils.data_utils import read_data


def image_input_shape(args):
    explicit = getattr(args, 'image_input_shape', None)
    if explicit is not None:
        shape = tuple(int(value) for value in explicit)
        if len(shape) != 3 or min(shape) < 1:
            raise ValueError('image_input_shape must be positive (channels, height, width)')
        return shape
    dataset = getattr(args, 'dataset', None)
    if not dataset:
        return None
    if getattr(args, '_image_shape_dataset', None) == dataset:
        return args._image_shape
    data = read_data(dataset, 0, is_train=True)['x']
    shape = None
    if len(data):
        try:
            sample_shape = np.asarray(data[0]).shape
        except (TypeError, ValueError):
            sample_shape = ()
        if len(sample_shape) == 3 and min(sample_shape) > 0:
            shape = tuple(int(value) for value in sample_shape)
    args._image_shape_dataset = dataset
    args._image_shape = shape
    return shape


def conv_flatten_dim(height, width, stages=2, channels=64):
    """Geometry of this framework's repeated Conv2d(k=5) + MaxPool2d(2)."""
    for _ in range(stages):
        height = (height - 4) // 2
        width = (width - 4) // 2
        if min(height, width) < 1:
            raise ValueError('Image is too small for the configured 5x5 convolution/pooling stages')
    return channels * height * width


def image_aware_spec(spec, args):
    """Supply measured dimensions at construction, including explicit legacy dims.

    Work on the local expression tree; never replace global model constructors
    or resize/reinitialize parameters during forward/optimizer updates.
    """
    tree = ast.parse(str(spec), mode='eval')
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        cnn = isinstance(call.func, ast.Name) and call.func.id in {'FedAvgCNN', 'CNN'}
        vit = (isinstance(call.func, ast.Attribute) and call.func.attr.startswith('vit_')
               and isinstance(call.func.value, ast.Attribute)
               and isinstance(call.func.value.value, ast.Name)
               and call.func.value.value.id == 'torchvision' and call.func.value.attr == 'models')
        if not (cnn or vit):
            continue
        shape = image_input_shape(args)
        if shape is None:
            continue
        if vit:
            if shape[1] != shape[2]:
                raise ValueError('torchvision ViT requires square images; provide a square image transform')
            name, value = 'image_size', ast.Constant(shape[1])
        else:
            name = 'input_shape'
            value = ast.Tuple(elts=[ast.Constant(size) for size in shape], ctx=ast.Load())
        call.keywords = [keyword for keyword in call.keywords if keyword.arg != name]
        call.keywords.append(ast.keyword(arg=name, value=value))
    return ast.fix_missing_locations(tree)


def adapt_image_channels(model, args):
    first = next((layer for layer in model.modules() if isinstance(layer, nn.Conv2d)), None)
    if first is None:
        return model
    shape = image_input_shape(args)
    if shape is None or first.in_channels == shape[0]:
        return model
    if first.groups != 1:
        raise ValueError('Cannot automatically adapt a grouped input convolution; specify a matching backbone')
    replacement = nn.Conv2d(shape[0], first.out_channels, first.kernel_size, first.stride,
                            first.padding, first.dilation, first.groups, first.bias is not None,
                            first.padding_mode).to(device=first.weight.device, dtype=first.weight.dtype)
    with torch.no_grad():
        # Preserve the response to an image repeated across all input channels.
        weight = first.weight.mean(1, keepdim=True) * (first.in_channels / shape[0])
        replacement.weight.copy_(weight.expand(-1, shape[0], -1, -1))
        if first.bias is not None:
            replacement.bias.copy_(first.bias)
            replacement.bias.requires_grad_(first.bias.requires_grad)
    replacement.weight.requires_grad_(first.weight.requires_grad)
    # Some backbones register their stem both as an attribute and in a Sequential.
    for parent in list(model.modules()):
        for name, child in list(parent._modules.items()):
            if child is first:
                setattr(parent, name, replacement)
    return model
