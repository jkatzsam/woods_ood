import sys
sys.path.insert(0, 'models')

from wrn import *

class WideResNet_SSND(nn.Module):
    def __init__(self, wrn):
        super(WideResNet_SSND, self).__init__()
        self.wrn = wrn
        self.num_classes = self.wrn.fc.out_features
        self.classifier = self.wrn.fc
        self.wrn.fc = nn.Identity()
        self.ood_fc1 = nn.Linear(self.classifier.in_features, 300)
        self.ood_fc2 = nn.Linear(300, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.wrn(x)
        x_class = self.classifier(x)

        x_ood = self.ood_fc1(x)
        x_ood = self.relu(x_ood)
        x_ood = self.ood_fc2(x_ood)

        x_all = torch.cat([x_class, x_ood], dim=1)

        return x_all