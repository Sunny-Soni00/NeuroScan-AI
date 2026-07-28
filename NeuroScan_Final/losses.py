import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridFocalDiceLoss(nn.Module):
    """
    Combines Dice Loss for shape alignment and Focal Loss for pixel imbalance.
    Expects raw logits as input (applies Sigmoid internally).
    """
    def __init__(self, lambda_dice=0.7, gamma_focal=2.0):
        super(HybridFocalDiceLoss, self).__init__()
        self.lambda_dice = lambda_dice
        self.gamma_focal = gamma_focal

    def dice_loss(self, inputs, targets, smooth=1e-6):
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

    def focal_loss(self, inputs, targets, alpha=0.8):
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        # Using binary_cross_entropy on sigmoid outputs
        BCE = F.binary_cross_entropy(inputs, targets, reduction='none')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP)**self.gamma_focal * BCE
        return focal_loss.mean()

    def forward(self, inputs, targets):
        return (self.lambda_dice * self.dice_loss(inputs, targets)) + \
               ((1 - self.lambda_dice) * self.focal_loss(inputs, targets))