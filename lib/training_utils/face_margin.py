import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_margin_module(margin_module_name, embeddings_size, class_nums, margin, s):
    if margin_module_name == "arcface":
        margin_module = ArcFace(embeddings_size, class_nums, margin, s)

    elif margin_module_name == "cosface":
        margin_module = CosFace(embeddings_size, class_nums, margin, s) 
    elif margin_module_name == "sphereface":
        margin_module = SphereFace(embeddings_size, class_nums, margin, s)

    elif margin_module_name == "softmax":
        margin_module = nn.Sequential()

    return margin_module


class SphereFace(nn.Module):
    """
    SphereFace loss function.
    Paper : https://arxiv.org/pdf/1704.08063.pdf
    
    The annealing optimization strategy for A-Softmax loss.
    """
    def __init__(self, embeddings_size, class_nums, margin, s, MAXLAMBDA=None, MINLAMBDA=None):
        super(SphereFace, self).__init__()

        self.identity_weights = nn.Parameter(torch.Tensor(embeddings_size, class_nums))

        self.margin = margin
        self.s = s

        # The hyperparameter of annealing optimization
        self.MAXLAMBDA = MAXLAMBDA
        self.MINLAMBDA = MINLAMBDA
        self.lamb = self.MAXLAMBDA
        # ============================================

        # Double angle formula.
        self.angle_formula = [
                    lambda x: x ** 0,                           # 0 * \theta
                    lambda x: x ** 1,                           # 1 * \theta
                    lambda x: 2 * x ** 2 - 1,                   # 2 * \theta
                    lambda x: 4 * x ** 3 - 3 * x,               # 3 * \theta
                    lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,      # 4 * \theta
                    lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x # 5 * \theta
                ]


    def forward(self, embeddings, label):
        identity_weights_norm = F.normalize(self.identity_weights, p=2)

        cos_theta = torch.mm(embeddings, identity_weights_norm)
        cos_theta = cos_theta.clamp(-1, 1)

        cos_m_theta = self.angle_formula[self.margin](cos_theta)

        # Calculate k for the phi which make m * \theta will not out of the range [0, pi]
        # Why we can calculate k by these formulate?
        theta = cos_theta.data.acos()
        k = (self.margin * theta / math.pi).floor()
        phi_theta = ((-1.0)**k) * cos_m_theta - 2*k

        x_norm = torch.norm(embeddings, 2, 1)

        one_hot = torch.zeros(cos_theta.size(), device=cos_theta.device)
        one_hot.scatter_(dim=1, index=label.view(-1, 1), src=1) # set the value to 1 according to the label

        # Annealing optimization. Annealing from Softmax to A-Softmax, which make training more stable.
        output = (one_hot * (phi_theta - cos_theta)) / (1 + self.lamb) + cos_theta # Utilize annealing for gt class. Otherwise utilize normalize cos_theta
        output *= x_norm.view(-1, 1)

        return output


class CosFace(nn.Module):
    """
    Cosface loss function.
    Paper : https://arxiv.org/pdf/1801.09414.pdf
    """
    def __init__(self, embeddings_size, class_nums, margin, s):
        super(CosFace, self).__init__()

        self.identity_weights = nn.Parameter(torch.Tensor(embeddings_size, class_nums))

        self.margin = margin
        self.s = s

    def forward(self, embeddings):
        identity_weights_norm = F.normalize(self.identity_weights, p=2)

        cos_theta = torch.mm(embeddings, identity_weights_norm)
        cos_theta = cos_theta.clamp(-1, 1)

        cos_theta_m = cos_theta - self.margin # cos(\theta) - m

        one_hot = torch.zeros(cos_theta.size(), device=cos_theta.device)
        one_hot.scatter_(dim=1, index=label.view(-1, 1), src=1) # set the value to 1 according to the label

        output = (one_hot * cos_theta_m) + ((1.0 - one_hot)*cos_theta) # Add margin to the gt class. Otherwise utilize normalize cos_theta
        output = output * self.s

        return output


class ArcFace(nn.Module):
    """
    Arcface loss function.
    Paper : https://arxiv.org/pdf/1801.07698.pdf
    """
    def __init__(self, embeddings_size, class_nums, margin, s):
        super(ArcFace, self).__init__()

        self.identity_weights = nn.Parameter(torch.Tensor(embeddings_size, class_nums))

        self.margin = margin
        self.threshold = math.cos(math.pi - margin)
        self.margin_cosface = math.sin(math.pi - margin) * margin

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

        self.mm = math.sin(math.pi - self.margin) * self.margin

        self.s = s

    def forward(self, embeddings, label):
        identity_weights_norm = F.normalize(self.identity_weights, p=2)

        cos_theta = torch.mm(embeddings, identity_weights_norm)
        cos_theta = cos_theta.clamp(-1, 1)

        cos_theta_2 = torch.pow(cos_theta, 2) # cos(\theta)^2
        sin_theta_2 = 1 - cos_theta_2 # sin(\theta)^2

        sin_theta = torch.sqrt(torch.clamp(sin_theta_2, 1e-9))

        # cos(a - b) = cos(a)cos(b) - sin(a)sin(b)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m # cos(\theta - m)


        # Control \theta + m range in [0, pi] ============================
        # when \theta + m >= pi, use cosface instead. Why?
        cos_theta_m = torch.where(cos_theta > self.threshold, cos_theta_m, cos_theta - self.mm)
        # ================================================================

        one_hot = torch.zeros(cos_theta.size(), device=cos_theta.device)
        idx = torch.arange(0, label.shape[0], dtype=torch.long)
        one_hot[torch.arange(0, label.shape[0], dtype=torch.long), label]  = 1

        output = (one_hot * cos_theta_m) + ((1.0 - one_hot)*cos_theta) # Add margin to the gt class. Otherwise utilize normalize cos_theta
        output = output * self.s

        return output

        
