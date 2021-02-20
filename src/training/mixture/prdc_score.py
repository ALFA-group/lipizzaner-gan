import logging

import numpy as np
import torch
import torchvision.transforms as transforms
from helpers.configuration_container import ConfigurationContainer
from prdc import compute_prdc
from torch import nn
from torchvision.models import vgg16
from training.mixture.fid_mnist import MNISTCnn
from training.mixture.score_calculator import ScoreCalculator


class ToRGB(object):
    def __call__(self, image):
        return image.convert("RGB")

class PRDCCalculator(ScoreCalculator):
    _logger = logging.getLogger(__name__)

    def __init__(
        self,
        imgs_original,
        batch_size=32,
        dims=64,
        n_samples=10000,
        cuda=True,
        verbose=False,
        nearest_k=5,
        use_random_vgg=True,
    ):
        """
        :param imgs_original: The original dataset, e.g. torcvision.datasets.CIFAR10
        :param batch_size: Batch size that will be used, 64 is recommended.
        :param cuda: If True, the GPU will be used.
        :param dims: Dimensionality of Emmbeding features to use. By default, uses MNIST features.
        :param n_samples: In the paper, min. 10k samples are suggested.
        :param verbose: Verbose logging
        """
        self.cc = ConfigurationContainer.instance()
        self.imgs_original = imgs_original
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.cuda = cuda
        self.verbose = verbose
        self.dataset = self.cc.settings["dataloader"]["dataset_name"]
        self.nearest_k = nearest_k
        self.dims = dims  # For MNIST the dimension of feature map is 10
        self.use_random_vgg = use_random_vgg

    def calculate(self, imgs, exact=True):
        """
        Calculates the Manifold Precision, Manifold Recall, Density and Coverage of two PyTorch datasets, which must have the same dimensions.

        :param imgs: PyTorch dataset containing the generated images. (Could be both grey or RGB images)
        :return: Harmonic mean of Density and Coverage and dict(PRDC)
        """
        model = None
        if self.dataset == "mnist" and not self.use_random_vgg:  # Gray dataset
            self.dims = 10
            model = MNISTCnn()
            model.load_state_dict(torch.load("./output/networks/mnist_cnn.pkl"))
        elif self.dataset not in ["unlabeled_gaussian_grid", "unlabeled_gaussian_circle"]:  # Other RGB dataset
            self.dims = 32
            model = vgg16()
            # Change the fc2 from (4096, 4096) to (4096, 64). Remove the last fc layer.
            model.classifier = nn.Sequential(*list(model.classifier.children())[:-4], nn.Linear(4096, self.dims))
            model.load_state_dict(torch.load(f"./output/networks/random_vgg16_{self.dims}.pth"))

        if model and self.cuda:
            model.cuda()

        real_act = self.get_activations(self.imgs_original, model)
        fake_act = self.get_activations(imgs, model, fake_batch=True)

        self._logger.info(f"vgg: {self.use_random_vgg}. real_act: {real_act.shape}, fake_act: {fake_act.shape}")

        metrics = compute_prdc(real_features=real_act, fake_features=fake_act, nearest_k=self.nearest_k)
        harmonic_mean = 2 * (metrics["density"] * metrics["coverage"]) / (metrics["density"] + metrics["coverage"])

        return harmonic_mean, metrics

    def get_activations(self, images, model, fake_batch=False):
        """Calculates the activations of the pool_3 layer for all images.

        Params:
        -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                         must lie between 0 and 1.
        -- model       : Instance of inception model
        Returns:
        -- A numpy array of dimension (num images, dims) that contains the
           activations of the given tensor when feeding inception with the
           query tensor.
        """
        assert len(images) >= self.n_samples, "Cannot draw enough samples from dataset"
        if fake_batch and self.use_random_vgg:
            transform = transforms.Compose([transforms.ToPILImage(), ToRGB(), transforms.Resize(224), transforms.ToTensor()])

        if model:
            model.eval()

            final_images = []

            for i in range(self.n_samples):
                # Reshape to 2D images as required by MNISTCnn class
                size = 224 if self.use_random_vgg else 28
                img = images[i]
                if fake_batch and self.use_random_vgg:
                    img = img.view(-1, 28, 28).cpu()
                    img = transform(img)
                img = img.view(-1, size, size)
                final_images.append(img)

            d0 = len(final_images)
            if self.batch_size > d0:
                print(("Warning: batch size is bigger than the data size. " "Setting batch size to data size"))
                self.batch_size = d0

            n_batches = d0 // self.batch_size
            n_used_imgs = n_batches * self.batch_size

            pred_arr = np.empty((n_used_imgs, self.dims))
            with torch.no_grad():
                for i in range(n_batches):
                    if self.verbose:
                        print(
                            "\rPropagating batch %d/%d" % (i + 1, n_batches),
                            end="",
                            flush=True,
                        )
                    start = i * self.batch_size
                    end = start + self.batch_size

                    batch = final_images[start:end]
                    batch = torch.stack(batch)
                    batch = torch.tensor(batch)
                    if self.cuda:
                        batch = batch.cuda()
                    else:
                        # .cpu() is required to convert to torch.FloatTensor because image
                        # might be generated using CUDA and in torch.cuda.FloatTensor
                        batch = batch.cpu()

                    if self.use_random_vgg:
                        pred = model(batch)
                    else:
                        pred = model(batch)[0]

                    pred_arr[start:end] = pred.cpu().data.numpy().reshape(self.batch_size, -1)

        else:
            pred_arr = [images[i] for i in range(self.n_samples)]

        if self.verbose:
            print(" done")

        return pred_arr

    @property
    def is_reversed(self):
        return False
