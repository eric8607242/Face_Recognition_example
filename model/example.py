import torch
import torch.nn as nn

from torchvision.models import resnet18

__all__ = [ "ExampleNet" ]

class ExampleNet(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.backbone = resnet18(pretrained=True)
        self.classifier = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self._forward_backbone(x)
        outputs = self.classifier(x)
        return outputs

    def _forward_backbone(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x


if __name__ == "__main__":
    from torchsummary import summary
    model = ExampleNet(n_classes=10575)
    model.eval()
    summary(model, (3, 112, 112), device="cpu")
