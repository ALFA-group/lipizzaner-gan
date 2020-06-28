"""
This file is used to evaluate the accuracy of a discriminator model obtained
upon training it in a Semi-Supervised environment. We load the network as well
as the parameters associated with the final layer following which the
BalancedLabelsBatchSampler is used to load a test dataset upon which the
discriminator model is tested. Many components are reused from the project such
as the DataLoader, ConfigurationContainer, etc enabling a very similar code
structure in this file as the rest of the project.
"""


import os

import torch
from data.balanced_labels_batch_sampler import BalancedLabelsBatchSampler
from helpers.configuration_container import ConfigurationContainer
from torchvision import datasets, transforms


def test(model, device, test_loader):
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 1, 28, 28)
            output = model.classification_layer(model.net(data))
            output = output.view(-1, 11)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('\nAccuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    cc = ConfigurationContainer.instance()
    cc.settings = {
        'network': {
            'name': 'ssgan_conv_mnist_28x28',
            'loss': 'celoss'
        },
        'trainer': {
            'params': {
                'score': {
                    'cuda': cuda
                }
            }
        }
    }

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)
            )
        ]
    )

    num_classes = 10
    batch_size = 100
    output_neurons = 784
    train = False
    dataset = datasets.MNIST(
        root=os.path.join('./output', 'data'),
        train=train,
        transform=transform,
        download=True)

    balanced_batch_sampler = BalancedLabelsBatchSampler(
        dataset,
        num_classes,
        batch_size,
        0.99
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        num_workers=0,
        batch_sampler=balanced_batch_sampler
    )

    network_factory = cc.create_instance(
        cc.settings['network']['name'],
        output_neurons,
        num_classes=num_classes
    )

    disc = network_factory.create_discriminator()
    checkpoint = torch.load(
        'discriminator.pkl',
        map_location=device
    )
    checkpoint_final_layer = torch.load(
        'discriminator_classification_layer.pkl',
        map_location=device
    )
    disc.net.load_state_dict(checkpoint)
    disc.net.eval()
    disc.classification_layer.load_state_dict(checkpoint_final_layer)
    disc.classification_layer.eval()
    test(disc, device, test_loader)


if __name__ == '__main__':
    main()
