import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch
import logging
import time

from networks.standalone_sgan.score_factory import ScoreCalculatorFactory

directory_or_file_name = int(time.time() * 1000)
logging.basicConfig(
    filename=f"networks/standalone_sgan/logs/log{directory_or_file_name}.log",
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
)
os.makedirs("networks/standalone_sgan/images", exist_ok=True)
os.makedirs(
    f"networks/standalone_sgan/images/output/{directory_or_file_name}", exist_ok=True,
)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument(
    "--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient",
)
parser.add_argument(
    "--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient",
)
parser.add_argument(
    "--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation",
)
parser.add_argument(
    "--latent_dim", type=int, default=100, help="dimensionality of the latent space",
)
parser.add_argument("--num_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument(
    "--sample_interval", type=int, default=400, help="interval between image sampling",
)
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False


def init_weights(m):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.complexity = 64
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(100, self.complexity * 8, 4, 1, 0),
            nn.BatchNorm2d(self.complexity * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.complexity * 8, self.complexity * 4, 4, 2, 1),
            nn.BatchNorm2d(self.complexity * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.complexity * 4, self.complexity * 2, 4, 2, 1),
            nn.BatchNorm2d(self.complexity * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.complexity * 2, self.complexity, 4, 2, 1),
            nn.BatchNorm2d(self.complexity),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.complexity, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, noise):
        img = self.conv_blocks(noise)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.complexity = 64
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, self.complexity, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.complexity, self.complexity * 2, 4, 2, 1),
            nn.BatchNorm2d(self.complexity * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.complexity * 2, self.complexity * 4, 4, 2, 1),
            nn.BatchNorm2d(self.complexity * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.complexity * 4, self.complexity * 8, 4, 2, 1),
            nn.BatchNorm2d(self.complexity * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Output layers

        self.adv_layer = nn.Sequential(nn.Conv2d(self.complexity * 8, 1, 4, 1, 0), nn.Sigmoid())

        self.aux_layer = nn.Sequential(nn.Conv2d(self.complexity * 8, opt.num_classes + 1, 4, 1, 0), nn.Softmax(),)

    def forward(self, img):
        out = self.conv_blocks(img)
        validity = self.adv_layer(out).view(-1)
        label = self.aux_layer(out).view(-1, 11)
        return validity, label


def to_pytorch_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def transpose_data(data):
    return data


def noise(batch_size, data_size):
    """
    Returns a variable with the dimensions (batch_size, data_size containing gaussian noise
    """
    shape = (batch_size,) + data_size if isinstance(data_size, tuple) else (batch_size, data_size)
    return to_pytorch_variable(torch.randn(shape))


# Loss functions
adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()


# Initialize weights
generator.apply(init_weights)
discriminator.apply(init_weights)


# Configure data loader
# os.makedirs("../../data/mnist", exist_ok=True)
os.makedirs("../../data/cifar", exist_ok=True)


def load():
    # Image processing

    # Dataset
    dataset = datasets.CIFAR10(
        root=os.path.join("./networks/standalone_sgan/images"), train=True, transform=transform(), download=True,
    )
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)


def transform():
    return transforms.Compose(
        [transforms.Resize(64), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
    )


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def save_images(images, shape, filename):
    save_image(denorm(images.data), filename)


dataloader = load()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        batch_size = imgs.shape[0]
        imgs = to_pytorch_variable(imgs)
        labels = to_pytorch_variable(transpose_data(labels))
        labels = torch.squeeze(labels)

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
        fake_aux_gt = Variable(LongTensor(batch_size).fill_(opt.num_classes), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = noise(batch_size, (100, 1, 1))
        # Generate a batch of images
        gen_imgs = generator(z)
        # Loss measures generator's ability to fool the discriminator
        validity, _ = discriminator(gen_imgs)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, real_aux = discriminator(real_imgs)
        real_acc = np.mean(real_pred.data.cpu().numpy())
        d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2

        # Loss for fake images
        fake_pred, fake_aux = discriminator(gen_imgs.detach())
        fake_acc = np.mean(fake_pred.data.cpu().numpy())
        d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, fake_aux_gt)) / 2

        # Total discriminator loss
        d_loss = d_real_loss + d_fake_loss

        # Calculate discriminator accuracy
        pred = real_aux.data.cpu().numpy()
        gt = labels.data.cpu().numpy()
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc_class: %d%%, acc_real: %d%%, acc_fake: %d%%] [G loss: %f]"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                d_loss.item(),
                100 * d_acc,
                100 * real_acc,
                100 * (1 - fake_acc),
                g_loss.item(),
            )
        )

    save_images(
        Variable(gen_imgs), (1, 28, 28), f"networks/standalone_sgan/images/output/{directory_or_file_name}/{epoch}.png",
    )
    logging.info(
        "[Epoch %d/%d] [D loss: %f, acc_class: %d%%, acc_real: %d%%, acc_fake: %d%%] [G loss: %f]"
        % (epoch, opt.n_epochs, d_loss.item(), 100 * d_acc, 100 * real_acc, 100 * (1 - fake_acc), g_loss.item(),)
    )

    score = float("-inf")
    calc = ScoreCalculatorFactory.create()
    logging.info("Score calculator: {}".format(type(calc).__name__))
    score = calc.calculate(gen_imgs)
    logging.info(f"Score: ({score[0]}, {score[1]})")
