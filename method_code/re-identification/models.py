import torch
import torchvision


class Resnet(torch.nn.Module):
    def __init__(self, model='resnet18', nclass=107, pretrained=True, freeze_base=False):
        super().__init__()
        model = getattr(torchvision.models, model)
        self.net = model(pretrained=pretrained)
        self.net.fc = torch.nn.Linear(self.net.fc.in_features, nclass)

        if freeze_base:
            for p in self.parameters():
                p.requires_grad = False
            for p in self.net.fc.parameters():
                p.requires_grad = True

    def forward(self, x):
        return self.net(x)


class ResnetEmbed(torch.nn.Module):
    def __init__(self, model='resnet18', pretrained=True, gray=False):
        super().__init__()
        model = getattr(torchvision.models, model)
        self.net = model(pretrained=pretrained)

        if gray:
            self.net.conv1.in_channels = 1
            p = torch.nn.Parameter(self.net.conv1.weight.mean(1, keepdim=True))
            self.net.conv1.weight = p

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        
        x = self.net.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = torch.nn.functional.normalize(x, p=2, dim=1)

        return x

