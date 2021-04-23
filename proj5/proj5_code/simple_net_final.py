import torch
import torch.nn as nn


class SimpleNetFinal(nn.Module):
    def __init__(self):
        """
        Constructor for SimpleNetFinal class to define the layers and loss
        function.

        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch's
        documention to understand what this means.
        """
        super(SimpleNetFinal, self).__init__()

        self.conv_layers = None
        self.fc_layers = None
        self.loss_criterion = None

        #######################################################################
        # Student code begins
        #######################################################################

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, stride=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(10, 60, kernel_size=5, stride=1),
            nn.BatchNorm2d(60),
            nn.Dropout(),
            nn.ReLU(),
            nn.Conv2d(60, 60, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(3)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(960, 100),
            nn.Linear(100, 15)
        )
        self.loss_criterion = nn.CrossEntropyLoss(reduction="mean")

        #######################################################################
        # Student code ends
        #######################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass with the network.

        Args:
            x: the (N,C,H,W) input images

        Returns:
            y: the (N,15) output (raw scores) of the net
        """
        model_output = None
        #######################################################################
        # Student code begins
        #######################################################################

        conv_features = self.conv_layers(x)
        flat_features = torch.flatten(conv_features, 1)
        model_output = self.fc_layers(flat_features)

        #######################################################################
        # Student code ends
        #######################################################################
        return model_output
