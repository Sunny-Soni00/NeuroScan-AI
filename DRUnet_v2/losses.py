import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridFocalDiceLoss(nn.Module):
    """
    Hybrid Loss combining Focal Loss and Dice Loss to handle:
    1. Class Imbalance (Focal)
    2. Spatial Overlap (Dice)
    """
    def __init__(self, lambda_dice=0.7, gamma_focal=2.0):
        super(HybridFocalDiceLoss, self).__init__()
        self.lambda_dice = lambda_dice
        self.gamma_focal = gamma_focal

    def dice_loss(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs)
        
        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

    def focal_loss(self, inputs, targets, alpha=0.8):
        inputs = torch.sigmoid(inputs)
        
        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Binary Cross Entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction='none')
        BCE_EXP = torch.exp(-BCE)
        
        # Focal calculation: alpha * (1-p)^gamma * BCE
        focal_loss = alpha * (1 - BCE_EXP)**self.gamma_focal * BCE
        return focal_loss.mean()

    def forward(self, inputs, targets):
        dl = self.dice_loss(inputs, targets)
        fl = self.focal_loss(inputs, targets)
        
        # Total Loss = λ * Dice + (1 - λ) * Focal
        return (self.lambda_dice * dl) + ((1 - self.lambda_dice) * fl)