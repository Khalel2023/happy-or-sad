import torch
import torch.nn as nn


class TinyVGG(torch.nn.Module):

    def __init__(self,input_shape,hidden_units,output_shape):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,out_channels=hidden_units,stride=1,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, stride=1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)

        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, stride=1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, stride=1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)

        )

        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(in_features=hidden_units*16*16,out_features=output_shape)
                                        )

    def forward(self,x):

        x = self.conv_block_1(x)
        x = self.conv_block_2(x)

        x = self.classifier(x)
        return x



