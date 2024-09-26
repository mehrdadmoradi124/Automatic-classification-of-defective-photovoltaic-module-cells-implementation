import torch
import torch.nn as nn
import torchvision.models as models

class ModifiedVGG19(nn.Module):
    def __init__(self):
        super(ModifiedVGG19, self).__init__()
        # Load pretrained VGG-19 model
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        
        # Remove the last two fully connected layers and add Global Average Pooling and new layers
        self.features = vgg19.features  # Keep the convolutional part of VGG-19
        self.fc = nn.Sequential(
            nn.Linear(512, 4096),  # First new FC layer
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2048),  # Second new FC layer
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1)  # Output layer for regression (predicting a single number)
        )
    
    def forward(self, x):
        # print("1",x.shape)
        x = self.features(x)
        # print("2",x.shape)
        x=torch.mean(x, dim=[2, 3])
        # print("3",x.shape)
        x = torch.flatten(x, 1)
        # print("4",x.shape)
        x = self.fc(x)
        x=x.view(16)
        # print("5",x.shape)
        return x

if __name__ == "__main__":
    model = ModifiedVGG19()
    print(model)