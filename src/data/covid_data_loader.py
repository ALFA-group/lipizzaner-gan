import logging

from helpers.configuration_container import ConfigurationContainer
from torchvision.datasets import ImageFolder

from torchvision.transforms import ToTensor, Compose, Resize, Grayscale
from torch.utils.data import Dataset
from data.data_loader import DataLoader
from torchvision.utils import save_image

from PIL import Image
import torch
from torch.autograd import Variable

from imblearn.over_sampling import SMOTE  # New packege

# Images size
WIDTH = 128
HEIGHT = 128


def smote_augmentation(tensor_list, labels_list, augmentation_times):
    # input list_tensors

    smote = SMOTE()

    for x in range(augmentation_times * len(tensor_list)):
        input_perturbation = Variable(torch.empty(tensor_list[0].shape).normal_(mean=0.5, std=0.001))
        tensor_list.append(input_perturbation)
        labels_list.append(-1)  # Label different to the original one

    stack = torch.stack(tensor_list)
    n_samples = stack.shape[0]
    colour_dimension = stack.shape[1]
    heigth = stack.shape[2]
    width = stack.shape[3]
    to_smote = stack.reshape(n_samples, colour_dimension * heigth * width)  # (n_samples, COLOUR * HEIGTH * WIDTH)

    sm = SMOTE()
    smoted_stack, smoted_labels = sm.fit_sample(to_smote, labels_list)

    augmented_tensor_list = []
    augmented_labels_list = []

    index = 0
    for x in smoted_stack:
        if smoted_labels[index] is not (-1):
            augmented_tensor_list.append(torch.from_numpy(x.reshape(colour_dimension, heigth, width)))
            augmented_labels_list.append(smoted_labels[index])
        index += 1

    return augmented_tensor_list, augmented_labels_list


class CovidDataLoader(DataLoader):
    def __init__(self, use_batch=True, batch_size=1, n_batches=0, shuffle=False):
        super().__init__(COVIDDataSet, use_batch, batch_size, n_batches, shuffle)

    @property
    def n_input_neurons(self):
        return WIDTH * HEIGHT

    @staticmethod
    def save_images(images, shape, filename):

        # img_view = data.view(num_images, 1, WIDTH, HEIGHT)
        img_view = images.view(images.size(0), 1, WIDTH, HEIGHT)
        # img_view = images.view(images)
        save_image(img_view, filename)

    @property
    def num_classes(self):
        return 0


class COVIDDataSet(Dataset):
    def __init__(self, **kwargs):

        self.cc = ConfigurationContainer.instance()
        settings = self.cc.settings["dataloader"]
        self.smote_augmentation_times = settings.get("smote_augmentation_times", 0)

        self.covid_type = settings.get("covid_type", "positive")

        self.use_batch = settings.get("use_batch", False)

        self.batch_size = settings.get("batch_size", None) if self.use_batch else None

        self._logger = logging.getLogger(__name__)

        # 1) Load data from file system
        transforms = [
            Grayscale(num_output_channels=1),
            Resize(size=[HEIGHT, WIDTH], interpolation=Image.NEAREST),
            ToTensor(),
        ]
        dataset = ImageFolder(root="data/datasets/covid-positive", transform=Compose(transforms))
        print(len(dataset))

        tensor_list = []
        labels_list = []
        for img in dataset:
            tensor_list.append(img[0])
            labels_list.append(img[1])

        self._logger.info("COVID-19 dataset samples: {}".format(len(tensor_list)))

        # 2) Applying SMOTE data augmentation
        if self.smote_augmentation_times is not None:
            tensor_list, labels_list = smote_augmentation(tensor_list, labels_list, self.smote_augmentation_times)

        self._logger.debug("Dataset size after SMOTE data augmentation: {}".format(len(tensor_list)))

        # 3) Remove incomplete batch
        if self.use_batch:
            reminder = len(tensor_list) % self.batch_size
            if reminder > 0:
                tensor_list = tensor_list[:-reminder]

        # 4) Merging all the data
        stacked_tensor = torch.stack(tensor_list)

        self._logger.debug("Final dataset shape: {}".format(stacked_tensor.shape))

        self.data = stacked_tensor
        self.labels = labels_list

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)
