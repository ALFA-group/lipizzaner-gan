import torch
import random
from networks.customized_loss.heuristic_loss import HeuristicLoss
from helpers.configuration_container import ConfigurationContainer


class MustangsLoss(torch.nn.Module):

    def __init__(self, applied_loss=None, name=None):
        super(MustangsLoss, self).__init__()
        self.losses_list = [torch.nn.BCELoss(), torch.nn.MSELoss(), HeuristicLoss()]

        # We add a new parameter to network named 'randomized'. If it is set to always it selects randomly a new loss
        # every minibatch in other case it selects randomly the loss function when the network is created
        cc = ConfigurationContainer.instance()
        if 'randomized' in cc.settings['network']:
            self.option_randomized = 'always' in cc.settings['network']['randomized']
            self.mustangs_always_random = self.option_randomized
        else:
            self.mustangs_always_random = False

        self.applied_loss = applied_loss
        if self.applied_loss is None:
            self.pick_random_lost()
        else:
            self.set_applied_loss(applied_loss)

        self.name = name

    def forward(self, input, target):
        # If we want to pick a random loss function for every minibatch
        if 'Generator' in self.name and self.mustangs_always_random:
            self.pick_random_lost()
        return self.applied_loss(input, target)

    def pick_random_lost(self):
        self.applied_loss = random.choice(self.losses_list)

    def get_applied_loss_name(self):
        return self.applied_loss.__class__.__name__

    def get_applied_loss(self):
        return self.applied_loss

    def set_applied_loss(self, loss):
        self.applied_loss = loss

    def set_network_name(self, name):
        self.name = name
        init = self.mustangs_always_random
        self.mustangs_always_random = False if ('Discriminator' in name) else self.option_randomized
