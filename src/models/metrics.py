import torch
import torch.nn.functional as F

def dice(preds, target, num_classes=3, epsilon=1e-6):
    pred_indices = torch.argmax(preds, dim=1) # (B, H, W)

    pred_one_hot = F.one_hot(pred_indices, num_classes=num_classes).permute(0, 3, 1, 2).float() # (B, C, H, W)
    target_one_hot = F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float() # (B, C, H, W)

    dice_scores = []
    
    for i in range(num_classes):
        pred_class = pred_one_hot[:, i]    # (B, H, W)
        target_class = target_one_hot[:, i] # (B, H, W)
        
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()
        
        dice = (2. * intersection + epsilon) / (union + epsilon)
        dice_scores.append(dice)
    
    return torch.tensor(dice_scores).mean()
