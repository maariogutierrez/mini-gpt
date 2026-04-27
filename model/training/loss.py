import torch.nn.functional as F


def cross_entropy_loss(logits, targets, ignore_index=0, reduction='mean'):
    # [B, T, V] -> [B*T, V]
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)
    
    # [B, T] -> [B*T]
    targets_flat = targets.view(-1)
    
    loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=ignore_index,
        reduction=reduction
    )
    
    return loss
