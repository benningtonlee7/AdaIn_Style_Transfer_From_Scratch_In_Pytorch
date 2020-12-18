import torch

def calc_mean_std(features, eps=1e-5):
    """
    :param features: dimension of features -> [batch_size, channel, height, weight]
    :param eps: a small value added to the variance to avoid divide-by-zero error.
    :return: features_mean, features_std: dimension of mean/std ->[batch_size, channel, 1, 1]
    """
    batch_size, num_channels = features.size()[:2]
    features_mean = features.view(batch_size, num_channels, -1).mean(dim=2).view(batch_size, num_channels, 1, 1)
    features_var = features.view(batch_size, num_channels, -1).var(dim=2) + eps
    features_std = features_var.sqrt().view(batch_size, num_channels, 1, 1)
    return features_mean, features_std

def adaptive_instance_normalization(content_features, style_features):
    """
    Adaptive Instance Normalization from  "Arbitrary Style Transfer in Real-time with
    Adaptive Instance Normalization, https://arxiv.org/abs/1703.06868"

    :param content_features: dimension -> [batch_size, channel, height, weight]
    :param style_features: dimension -> [batch_size, channel, height, weight]
    :return: normalized_features dimension -> [batch_size, channel, height, weight]
    """
    content_mean, content_std = calc_mean_std(content_features)
    style_mean, style_std = calc_mean_std(style_features)
    normalized_features = style_std * (content_features - content_mean) / content_std + style_mean

    return normalized_features

def denormalize(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res

def learning_rate_decay(lr, decay, iteration):
    return lr / (1.0 + decay * iteration)

def calc_flatten_mean_std(feat):
    flatten = feat.view(3, -1)
    mean = flatten.mean(dim=-1, keepdim=True)
    std = flatten.std(dim=-1, keepdim=True)
    return flatten, mean, std

def matrix_sqrt(mat):
    U, D, V = torch.svd(mat)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())

# From https://arxiv.org/abs/1611.07865, "Controlling Perceptual Factors in Neural Style Transfer"
def color_control(original, target):
    flatten_o, mean_o, std_o = calc_flatten_mean_std(original)
    normalized_o = (flatten_o - mean_o.expand_as(flatten_o)) / std_o.expand_as(flatten_o)
    cov_eye_o = torch.mm(normalized_o, normalized_o.t()) + torch.eye(3)

    flatten_t, mean_t, std_t = calc_flatten_mean_std(target)
    normalized_t = (flatten_t - mean_o.expand_as(flatten_t)) / std_o.expand_as(flatten_t)
    cov_eye_t = torch.mm(normalized_t, normalized_t.t()) + torch.eye(3)

    normalized_transfer = torch.mm(matrix_sqrt(cov_eye_t), torch.mm(torch.inverse(matrix_sqrt(cov_eye_o)), normalized_o))
    original_transfer = normalized_transfer * std_t.expand_as(normalized_o) + mean_t.expand_as(normalized_o)

    return original_transfer.view(original.size())

