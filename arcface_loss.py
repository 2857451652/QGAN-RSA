import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ArcFaceLoss(nn.Module):
    def __init__(self, margin=0.1, scale=64):
        super(ArcFaceLoss, self).__init__()
        self.margin = margin
        self.scale = scale
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.mm = self.sin_m * self.margin  # cos(m) - 1
        self.threshold = math.cos(math.pi - self.margin)

    def forward(self, x, c, labels, c_mask=None):
        # Normalize the input embeddings
        x = F.normalize(x)

        # Normalize the candidates embeddings
        c = F.normalize(c, p=2, dim=-1)

        # Compute the cosine similarity between the embeddings and the weights
        # cos_theta = torch.matmul(x, c.transpose(0, 1))
        cos_theta = torch.matmul(x.unsqueeze(1), c.transpose(-2, -1)).squeeze(1)

        # Compute the sine and cosine of (theta + m)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        # Apply the margin to the cosine similarity
        cos_theta_m = torch.where(cos_theta > self.threshold, cos_theta_m, cos_theta - self.mm)

        # Create a one-hot encoding of the labels
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Compute the final output
        output = (one_hot * cos_theta_m) + ((1.0 - one_hot) * cos_theta)
        output *= self.scale

        loss = F.cross_entropy(output, labels)
        return loss


