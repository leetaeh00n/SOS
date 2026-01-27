import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparationHead(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(h))).squeeze(-1)

def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss

class WRN_VOG(nn.Module):
    def __init__(self, feature_sizes, num_channels, interm_dim=128):
        super(WRN_VOG, self).__init__()
        self.gap1 = nn.AvgPool2d(kernel_size=feature_sizes[0])
        self.gap2 = nn.AvgPool2d(kernel_size=feature_sizes[1])
        self.gap3 = nn.AvgPool2d(kernel_size=feature_sizes[2])

        self.fc1 = nn.Linear(num_channels[0], interm_dim)
        self.fc2 = nn.Linear(num_channels[1], interm_dim)
        self.fc3 = nn.Linear(num_channels[2], interm_dim)

        self.out_proj = nn.Linear(3 * interm_dim, 128)

    def forward(self, features):
        x1 = self.gap1(features[0]).view(features[0].size(0), -1)
        x1 = F.relu(self.fc1(x1))
        x2 = self.gap2(features[1]).view(features[1].size(0), -1)
        x2 = F.relu(self.fc2(x2))
        x3 = self.gap3(features[2]).view(features[2].size(0), -1)
        x3 = F.relu(self.fc3(x3))
        out = torch.cat([x1, x2, x3], dim=1)
        return self.out_proj(out)  # [B, 128]