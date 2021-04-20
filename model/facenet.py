import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class FaceNet(nn.Module):

    def __init__(self, in_channels, n_features):
        super().__init__()
        # Pretrained backbone
        self.feature = resnet18(pretrained=True)
        self.feature.fc = None
        # Embedding Layer
        self.embedding = nn.Linear(512, n_features)

    def forward(self, x, normed=True):
        x = self._forward_feature(x)
        x = self.embedding(x)
        if normed:
            x = F.normalize(x, p=2, dim=1)
        return x

    def _forward_feature(self, x):
        x = self.feature.conv1(x)
        x = self.feature.bn1(x)
        x = self.feature.relu(x)
        x = self.feature.maxpool(x)
        x = self.feature.layer1(x)
        x = self.feature.layer2(x)
        x = self.feature.layer3(x)
        x = self.feature.layer4(x)
        x = self.feature.avgpool(x)
        x = torch.flatten(x, 1)
        return x


if __name__ == "__main__":
    model = FaceNet(in_channels=3, n_features=128)
    x = torch.rand(32, 3, 112, 112)
    output = model(x)
    print(output.shape)
