import math
import torch

def make_to_onehot_by_scatter(input_tensor: torch.Tensor, num_classes: int):
    """
    Args:
        input_tensor: batch_size, num_channel
        num_classes:
    Returns: batch_size, num_channel, num_classes
    """
    one_hot = torch.zeros(size=(math.prod(input_tensor.shape), num_classes)).to(input_tensor.device)
    one_hot.scatter_(dim=1, index=input_tensor.reshape(-1, 1), value=1)
    one_hot = one_hot.reshape(*input_tensor.shape, num_classes)
    return one_hot
