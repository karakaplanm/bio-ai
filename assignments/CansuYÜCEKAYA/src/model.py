import torch.nn as nn

class BacteriaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*32*32,256),
            nn.ReLU(),
            nn.Linear(256,3)
        )
    def forward(self,x):
        x=self.conv(x)
        x=self.fc(x)
        return x
