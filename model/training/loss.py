import torch
import torch.nn.functional as F


def cross_entropy_loss(logits, targets, ignore_index=0, reduction='mean'):
    """
    Compute cross-entropy loss on logits vs. targets with padding token support.
    
    Args:
        logits (torch.Tensor): Model output logits of shape [B, T, V]
            where B is batch size, T is sequence length, V is vocabulary size
        targets (torch.Tensor): Target token IDs of shape [B, T]
        ignore_index (int): Token index to ignore (typically padding token, default: 0)
        reduction (str): 'mean', 'sum', or 'none' for loss reduction
    
    Returns:
        torch.Tensor: Scalar loss value (if reduction != 'none')
    
    Example:
        >>> logits = torch.randn(2, 10, 1000)  # [B=2, T=10, V=1000]
        >>> targets = torch.randint(0, 1000, (2, 10))  # [B=2, T=10]
        >>> loss = cross_entropy_loss(logits, targets, ignore_index=0)
    """
    # Reshape logits: [B, T, V] -> [B*T, V]
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)
    
    # Reshape targets: [B, T] -> [B*T]
    targets_flat = targets.view(-1)
    
    # Compute cross-entropy loss with ignore_index for padding tokens
    loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=ignore_index,
        reduction=reduction
    )
    
    return loss
