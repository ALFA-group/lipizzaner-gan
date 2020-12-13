import logging

import numpy as np
import torch
from helpers.configuration_container import ConfigurationContainer
from prdc import compute_prdc
from torch.autograd import Variable
from training.mixture.fid_mnist import MNISTCnn
from training.mixture.score_calculator import ScoreCalculator


class PRDCCalculator(ScoreCalculator):
    _logger = logging.getLogger(__name__)

    def __init__(
        self,
        imgs_original,
        batch_size=64,
        dims=10,
        n_samples=10000,
        cuda=True,
        verbose=False,
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
        self.network = self.cc.settings["network"]["name"]
        self.dims = dims  # For MNIST the dimension of feature map is 10

    def calculate(self, imgs, exact=True, nearest_k=10):
        """
        Calculates the Manifold Precision, Manifold Recall, Density and Coverage of two PyTorch datasets, which must have the same dimensions.

        :param imgs: PyTorch dataset containing the generated images. (Could be both grey or RGB images)
        :return: Harmonic mean of Density and Coverage and dict(PRDC)
        """
        model = None
        if self.dataset == "mnist":  # Gray dataset
            model = MNISTCnn()
            model.load_state_dict(torch.load("./output/networks/mnist_cnn.pkl"))
        else:  # Other RGB dataset
            # TODO: Add Dynamic definition of ConvNet.
            #       With matching input size to dataset and output size to self.dims.
            raise Exception('Datset {} is not supported. Use "MNIST".'.format(self.dataset))

        if self.cuda:
            model.cuda()

        real_act = self.get_activations(self.imgs_original, model)
        fake_act = self.get_activations(imgs, model)

        metrics = compute_prdc(real_features=real_act, fake_features=fake_act, nearest_k=nearest_k)
        harmonic_mean = 2 * (metrics["density"] * metrics["coverage"]) / (metrics["density"] + metrics["coverage"])

        return harmonic_mean, metrics

    def get_activations(self, images, model):
        """Calculates the activations of the pool_3 layer for all images.

        Params:
        -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                         must lie between 0 and 1.
        -- model       : Instance of inception model
        -- batch_size  : the images numpy array is split into batches with
                         batch size batch_size. A reasonable batch size depends
                         on the hardware.
        -- dims        : Dimensionality of features returned by Inception

        Returns:
        -- A numpy array of dimension (num images, dims) that contains the
           activations of the given tensor when feeding inception with the
           query tensor.
        """
        model.eval()

        # Reshape to 2D images as required by MNISTCnn class
        images = [img.view(-1, 28, 28) for img in images]

        d0 = len(images)
        if self.batch_size > d0:
            print(("Warning: batch size is bigger than the data size. " "Setting batch size to data size"))
            self.batch_size = d0

        n_batches = d0 // self.batch_size
        n_used_imgs = n_batches * self.batch_size

        pred_arr = np.empty((n_used_imgs, self.dims))
        for i in range(n_batches):
            if self.verbose:
                print(
                    "\rPropagating batch %d/%d" % (i + 1, n_batches),
                    end="",
                    flush=True,
                )
            start = i * self.batch_size
            end = start + self.batch_size

            batch = torch.stack(images[start:end])
            batch = Variable(batch, volatile=True)
            if self.cuda:
                batch = batch.cuda()
            else:
                # .cpu() is required to convert to torch.FloatTensor because image
                # might be generated using CUDA and in torch.cuda.FloatTensor
                batch = batch.cpu()

            pred = model(batch)[0]

            pred_arr[start:end] = pred.cpu().data.numpy().reshape(self.batch_size, -1)

        if self.verbose:
            print(" done")

        return pred_arr

    @property
    def is_reversed(self):
        return False
