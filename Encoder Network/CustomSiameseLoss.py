import torch.nn as nn
import torch.nn.functional as F

class customSiameseLossFunction(nn.Module):
    def __init__(self, gamma_p, gamma_d, gamma_n, margin=1.0):
        super(customSiameseLossFunction, self).__init__()
        self.gamma_p = gamma_p
        self.gamma_d = gamma_d
        self.gamma_n = gamma_n
        self.margin = margin
        self.dist = nn.PairwiseDistance(p=2)
    
    def forward(self, y_anchor, y_pos, y_neg, neg_type):
        # Calculate base distances
        d_ap = self.dist(y_anchor, y_pos) 
        d_an = self.dist(y_anchor, y_neg)

        # Calculate weighted negative distance based on type
        # If neg_type=0 (defective), use gamma_d
        # If neg_type=1 (fraud/new object), use gamma_n
        weighted_d_an = (1 - neg_type) * self.gamma_d * d_an + neg_type * self.gamma_n * d_an

        # The Triple Logic: Weighted Positive - Weighted Negative + Margin
        # We want (gamma_p * d_ap) to be smaller than (weighted_d_an) by at least 'margin'
        loss = F.relu(self.gamma_p * d_ap - weighted_d_an + self.margin)

        return loss.mean()