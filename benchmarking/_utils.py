import torch
from copy import deepcopy

def same_device(device1: torch.device, device2: torch.device) -> bool:
    """Return True if two devices are effectively the same."""
    if device1.type != device2.type:
        return False
    return (device1.index == device2.index) or (device1.index is None or device2.index is None)


def clone_grads(tensor_or_tensors):
    """Clone gradients from tensor(s) that require grad.
    
    Args:
        tensor_or_tensors: A tensor, dict, or iterable of items that may be tensors
        
    Returns:
        List of cloned gradients for any tensors that required grad
    """
    # Handle single tensor case
    if isinstance(tensor_or_tensors, torch.Tensor):
        if tensor_or_tensors.requires_grad and tensor_or_tensors.grad is not None:
            return tensor_or_tensors.grad.detach().clone()
        return None
        
    # Handle dict case
    if isinstance(tensor_or_tensors, dict):
        grads = {}
        for key, item in tensor_or_tensors.items():
            if isinstance(item, torch.Tensor) and item.requires_grad:
                if item.grad is not None:
                    grads[key] = item.grad.detach().clone()
                else:
                    grads[key] = None
        return grads
        
    # Handle iterable case
    grads = []
    for item in tensor_or_tensors:
        if isinstance(item, torch.Tensor) and item.requires_grad:
            if item.grad is not None:
                grads.append(item.grad.detach().clone())
            else:
                grads.append(None)
    return grads


def clear_grads(tensor_or_tensors):
    """Clear gradients from tensor(s) that require grad.
    
    Args:
        tensor_or_tensors: A tensor or iterable of items that may be tensors
    """
    # Handle single tensor case
    if isinstance(tensor_or_tensors, torch.Tensor):
        if tensor_or_tensors.requires_grad:
            tensor_or_tensors.grad = None
        return
    # Handle iterable/dict case
    if isinstance(tensor_or_tensors, dict):
        for item in tensor_or_tensors.values():
            if isinstance(item, torch.Tensor) and item.requires_grad:
                item.grad = None
    else:
        for item in tensor_or_tensors:
            if isinstance(item, torch.Tensor) and item.requires_grad:
                item.grad = None

