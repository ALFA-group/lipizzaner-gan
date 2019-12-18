import os

import logging
import pathlib

import requests
from torch.autograd import Variable
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
from torchvision.utils import save_image

from data.data_loader import DataLoader
from helpers.configuration_container import ConfigurationContainer
from helpers.pytorch_helpers import denorm


class CelebADataLoader(DataLoader):

    def __init__(self, use_batch=True, batch_size=100, n_batches=0, shuffle=False):
        super().__init__(CelebADataSet, use_batch, batch_size, n_batches, shuffle)

    @property
    def n_input_neurons(self):
        return 3072

    @property
    def num_classes(self):
        return None

    def transform(self):
        return transforms.Compose([transforms.Resize([64,64]), transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def save_images(self, images, shape, filename):
        data = images.data if isinstance(images, Variable) else images
        save_image(denorm(data), filename)

    def transpose_data(self, data):
        return data


class CelebADataSet(ImageFolder):

    _logger = logging.getLogger(__name__)

    file_id = '0B7EVK8r0v71pZjFTYXZWM3FlRnM'
    filename = 'img_align_celeba.zip'
    base_folder = 'img_align_celeba'

    def __init__(self, root, transform=None, target_transform=None, download=True, **kwargs):
        target_dir = os.path.join(root, self.base_folder)
        try:
            if download:
                self.download(target_dir)
            super().__init__(target_dir, transform, target_transform)
        except Exception as ex:
            self._logger.critical("An error occured while trying to download CelebA: {}".format(ex))
            raise ex

    def download(self, target_dir):
        import zipfile

        if self._already_downloaded(target_dir):
            self._logger.info('CelebA dataset is already downloaded and extracted.')
            return

        self._logger.info('CelebA dataset has to be downloaded, this may take some time...')
        pathlib.Path(target_dir).mkdir(parents=True, exist_ok=True)

        self._download_file_from_google_drive(self.file_id, os.path.join(target_dir, self.filename))
        self._logger.info('Download finished, extracting...')

        with zipfile.ZipFile(os.path.join(target_dir, self.filename), "r") as zip_ref:
            zip_ref.extractall(target_dir)

        self._logger.info('File successfully extracted.')

    @staticmethod
    def _already_downloaded(target_dir):
        return os.path.isdir(target_dir) and os.listdir(target_dir)

    @staticmethod
    def _download_file_from_google_drive(id, destination):
        def get_confirm_token(response):
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    return value

            return None

        def save_response_content(response, destination):
            CHUNK_SIZE = 32768

            with open(destination, "wb") as f:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)

        URL = "https://docs.google.com/uc?export=download"

        session = requests.Session()

        response = session.get(URL, params={'id': id}, stream=True)
        token = get_confirm_token(response)

        if token:
            params = {'id': id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        save_response_content(response, destination)
