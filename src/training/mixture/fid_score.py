#!/usr/bin/env python3
"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

Adapted from https://github.com/mseitzer/pytorch-fid

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import logging

import numpy as np
import torch
from scipy import linalg
from torch.autograd import Variable
from torch.nn.functional import adaptive_avg_pool2d

from helpers.configuration_container import ConfigurationContainer
from training.mixture.fid_mnist import MNISTCnn
from training.mixture.fid_mnist_conv import MNISTConvCnn
from training.mixture.fid_inception import InceptionV3
from training.mixture.score_calculator import ScoreCalculator


class FIDCalculator(ScoreCalculator):

    _logger = logging.getLogger(__name__)

    def __init__(self, imgs_original, batch_size=64, dims=2048, n_samples=10000, cuda=True, verbose=False):
        """
        :param imgs_original: The original dataset, e.g. torcvision.datasets.CIFAR10
        :param batch_size: Batch size that will be used, 64 is recommended.
        :param cuda: If True, the GPU will be used.
        :param dims: Dimensionality of Inception features to use. By default, uses pool3 features.
        :param n_samples: In the paper, min. 10k samples are suggested.
        :param verbose: Verbose logging
        """
        self.cc = ConfigurationContainer.instance()
        self.imgs_original = imgs_original
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.cuda = cuda
        self.verbose = verbose
        if self.cc.settings['dataloader']['dataset_name'] == 'mnist' or self.cc.settings['dataloader']['dataset_name'] == 'mnist_fashion':
            self.dims = 10    # For MNIST the dimension of feature map is 10
        else:
            self.dims = dims

    def calculate(self, imgs, exact=True):
        """
        Calculates the Fréchet Inception Distance of two PyTorch datasets, which must have the same dimensions.

        :param imgs: PyTorch dataset containing the generated images. (Could be both grey or RGB images)
        :param exact: Currently has no effect for FID.
        :return: FID and TVD
        """
        model = None
        if self.cc.settings['dataloader']['dataset_name'] == 'mnist':    # Gray dataset
            if self.cc.settings['network']['name'] == 'ssgan_convolutional_mnist':
                model = MNISTConvCnn()
                model.load_state_dict(torch.load('./output/networks/mnist_conv_cnn.pt'))
            else:
                model = MNISTCnn()
                model.load_state_dict(torch.load('./output/networks/mnist_cnn.pkl'))
            compute_label_freqs = True
        elif self.cc.settings['dataloader']['dataset_name'] == 'mnist_fashion':
            model = MNISTCnn()
            model.load_state_dict(torch.load('./output/networks/fashion_mnist_cnn.pt'))
            compute_label_freqs = True
        else:    # Other RGB dataset
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
            model = InceptionV3([block_idx])
            compute_label_freqs = False

        if self.cuda:
            model.cuda()
        m1, s1, freq1 = self._compute_statistics_of_path(self.imgs_original, model, compute_label_freqs)
        m2, s2, freq2 = self._compute_statistics_of_path(imgs, model, compute_label_freqs)

        tvd = 0.5 * sum(abs(f1 - f2) for f1, f2 in zip(freq1, freq2))

        return abs(self.calculate_frechet_distance(m1, s1, m2, s2)), tvd

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

        d0 = len(images)
        if self.batch_size > d0:
            print(('Warning: batch size is bigger than the data size. '
                   'Setting batch size to data size'))
            self.batch_size = d0

        n_batches = d0 // self.batch_size
        n_used_imgs = n_batches * self.batch_size

        pred_arr = np.empty((n_used_imgs, self.dims))
        for i in range(n_batches):
            if self.verbose:
                print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                      end='', flush=True)
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

            if self.cc.settings['dataloader']['dataset_name'] != 'mnist' \
                and self.cc.settings['dataloader']['dataset_name'] != 'mnist_fashion':
                # If model output is not scalar, apply global spatial average pooling.
                # This happens if you choose a dimensionality not equal 2048.
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred_arr[start:end] = pred.cpu().data.numpy().reshape(self.batch_size, -1)

        if self.verbose:
            print(' done')

        return pred_arr

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representive data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representive data set.

        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                # raise ValueError('Imaginary component {}'.format(m))

                # Temporarily ignore the error, and log it for reminder that imaginary component appears
                self._logger.info('ValueError (but ignored): Imaginary component {}'.format(m))

            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)

    def get_frequencies_of_activations(self, activations):
        frequencies = 10 * [0]
        for ac in activations:
            frequencies[list(ac).index(max(list(ac)))] += 1
        frequencies = [f / len(activations) for f in frequencies]
        return frequencies

    def calculate_activation_statistics(self, images, model, compute_label_freqs=False):
        """Calculation of the statistics used by the FID and TVD.
        Params:
        -- images      : Numpy array of dimension (n_images, 3, hi, wi). The values
                         must lie between 0 and 1.
        -- model       : Instance of inception model
        -- batch_size  : The images numpy array is split into batches with
                         batch size batch_size. A reasonable batch size
                         depends on the hardware.
        -- dims        : Dimensionality of features returned by Inception
        -- cuda        : If set to True, use GPU
        -- verbose     : If set to True and parameter out_step is given, the
                         number of calculated batches is reported.
        Returns:
        -- mu    : The mean over samples of the activations of the pool_3 layer of
                   the inception model.
        -- sigma : The covariance matrix of the activations of the pool_3 layer of
                   the inception model.
        """
        act = self.get_activations(images, model)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)

        frequencies = self.get_frequencies_of_activations(act) if compute_label_freqs else []

        return mu, sigma, frequencies

    def _compute_statistics_of_path(self, dataset, model, compute_label_freqs=False):
        imgs = []
        assert len(dataset) >= self.n_samples, 'Cannot draw enough samples from dataset'

        for i in range(self.n_samples):
            img = dataset[i]
            if self.cc.settings['dataloader']['dataset_name'] == 'mnist':
                if self.cc.settings['network']['name'] == 'ssgan_convolutional_mnist':
                    img = img.view(-1, 64, 64)
                else:
                    img = img.view(-1, 28, 28)
            if self.cc.settings['dataloader']['dataset_name'] == 'mnist_fashion':
                # Reshape to 2D images as required by MNISTCnn class
                img = img.view(-1, 28, 28)

            imgs.append(img)

        return self.calculate_activation_statistics(imgs, model, compute_label_freqs)

    @property
    def is_reversed(self):
        return True
