import torch
import torch.nn as nn

from torchvision.models import resnet18


class MyResNet18(nn.Module):
    def __init__(self):
        """
        Initializes network layers.

        Note: Do not forget to freeze the layers of ResNet except the last one.
        Note: Use 'mean' reduction in the loss_criterion. Read Pytorch's
            documention to understand what this means.

        Download pretrained ResNet using pytorch's API.

        Hint: see the import statements
        """
        super().__init__()

        self.conv_layers = None
        self.fc_layers = None
        self.loss_criterion = None

        #######################################################################
        # Student code begins
        #######################################################################

        model = resnet18(pretrained=True)
        self.conv_layers = nn.Sequential(
            *list(model.children())[0:-1]
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 15)
        )
        for param in self.conv_layers.parameters():
            param.requires_grad = False
        self.loss_criterion = nn.CrossEntropyLoss(reduction="mean")

        #######################################################################
        # Student code ends
        #######################################################################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass with the net, duplicating grayscale channel to
        3 channels.

        Args:
            x: tensor of shape (N,C,H,W) representing input batch of images

        Returns:
            y: tensor of shape (N,num_classes) representing the output
                (raw scores) of the network. Note: we set num_classes=15
        """
        model_output = None
        x = x.repeat(1, 3, 1, 1)  # as ResNet accepts 3-channel color images
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
