import torch
import random
from networks.customized_loss.heuristic_loss import HeuristicLoss
from helpers.configuration_container import ConfigurationContainer


class MustangsLoss(torch.nn.Module):
    def __init__(self, applied_loss=None, name=None):
        super(MustangsLoss, self).__init__()
        self.losses_list = [
            torch.nn.BCELoss(),
            torch.nn.MSELoss(),
            HeuristicLoss(),
        ]

        # We add a new parameter to network named 'randomized'. If it is set to always it selects randomly a new loss
        # every minibatch in other case it selects randomly the loss function when the network is created
        cc = ConfigurationContainer.instance()
        self.mustangs_always_random_init_value = "always" == cc.settings["network"].get("randomized", "False")
        self.mustangs_always_random = self.mustangs_always_random_init_value

        self.applied_loss = applied_loss
        if self.applied_loss is None:
            self.pick_random_loss()
        else:
            self.set_applied_loss(applied_loss)

        self.name = name

    def forward(self, input, target):
        # If we want to pick a random loss function for every minibatch
        if "Generator" in self.name and self.mustangs_always_random:
            self.pick_random_loss()
        return self.applied_loss(input, target)

    def pick_random_loss(self):
        self.applied_loss = random.choice(self.losses_list)

    def get_applied_loss_name(self):
        return self.applied_loss.__class__.__name__

    def get_applied_loss(self):
        return self.applied_loss

    def set_applied_loss(self, loss):
        self.applied_loss = loss

    def set_network_name(self, name):
        self.name = name
        self.mustangs_always_random = False if ("Discriminator" in name) else self.mustangs_always_random_init_value
