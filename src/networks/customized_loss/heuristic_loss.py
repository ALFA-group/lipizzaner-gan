import torch


class HeuristicLoss(torch.nn.Module):
    def __init__(self):
        super(HeuristicLoss, self).__init__()

    def forward(self, input, target):
        return -0.5 * torch.mean(torch.log(input), dim=0)
