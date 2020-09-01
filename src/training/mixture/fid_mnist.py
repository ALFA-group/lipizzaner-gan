import torch.nn as nn
import torch.nn.functional as F


class MNISTCnn(nn.Module):
    """ Simple CNN trained on MNIST for returning feature maps
    To calculate FID for MNIST, we do not use InceptionV3, but instead
    manually train a classifier on MNIST dataset
    (Referred from: https://openreview.net/forum?id=Hy7fDog0b)
    """

    def __init__(self, requires_grad=False):
        """ Build simple CNN """
        super(MNISTCnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx1x28x28. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable outputted by fc2
        """
        assert len(inp.shape) == 4 and inp.shape[1] == 1 and inp.shape[2] == 28 and inp.shape[3] == 28
        outp = []
        x = inp

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=False)
        x = self.fc2(x)
        outp.append(x)  # Store output of fc2

        return outp
