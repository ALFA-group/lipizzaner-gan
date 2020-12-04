import logging

from helpers.configuration_container import ConfigurationContainer
from torchvision.datasets import ImageFolder

from torchvision.transforms import ToTensor, Compose, Resize, Grayscale, Normalize
from torch.utils.data import Dataset
from data.data_loader import DataLoader
from torchvision.utils import save_image

from PIL import Image
import torch
from torch.autograd import Variable

from imblearn.over_sampling import SMOTE

WIDTH = 28
HEIGHT = 28




class HMNISTDataLoader(DataLoader):

    def __init__(self, use_batch=True, batch_size=1, n_batches=0, shuffle=False):
        super().__init__(HMNISTDataSet, use_batch, batch_size, n_batches, shuffle)

    @property
    def n_input_neurons(self):
        return WIDTH*HEIGHT

    @staticmethod
    def save_images(images, shape, filename):

        # img_view = data.view(num_images, 1, WIDTH, HEIGHT)
        img_view = images.view(images.size(0), 1, WIDTH, HEIGHT)
        # img_view = images.view(images)
        save_image(img_view, filename)


class HMNISTDataSet(Dataset):

    def __init__(self, **kwargs):

        self.cc = ConfigurationContainer.instance()
        settings = self.cc.settings['dataloader']


        self.use_batch = settings.get('use_batch', False)

        self.batch_size = settings.get('batch_size', None) if self.use_batch else None

        self._logger = logging.getLogger(__name__)

        # 1) CARGAR IMAGENES DESDE EL FILESYSTEM

        # Se cargan las imagenes en una lista de tuplas <tensor,int> donde:
        # tensor.shape = (1, HEIGHT, WIDTH)
        # int es el indice de la clase asociada a dicho tensor

        transforms = [Grayscale(num_output_channels=1), Resize(size=[HEIGHT, WIDTH], interpolation=Image.NEAREST), ToTensor(), Normalize(mean=(0.5,), std=(0.5,)), ]
        #transforms = [Resize(size=[HEIGHT, WIDTH], interpolation=Image.NEAREST), ToTensor()]
        dataset = ImageFolder(root="data/datasets/base_dir/train_dir", transform=Compose(transforms))
        #dataset = ImageFolder(root="data/datasets/base_dir/train_dir")
        print(len(dataset))

        # Se separan las tuplas en lista de tensores y lista de labels
        tensor_list = []
        labels_list = []
        for img in dataset:
            tensor_list.append(img[0])
            labels_list.append(img[1])

        print("Original dataset size: " + str(len(tensor_list)))



        # 3) REMOVER BATCH INCOMPLETO

        # Remuevo los ultimos elementos que no completan un batch
        if self.use_batch:
            reminder = len(tensor_list) % self.batch_size
            if reminder > 0:
                tensor_list = tensor_list[:-reminder]

        # 4) UNIFICAR LISTA EN TENSOR UNICO
        # Se conVierte la lista de tensores en un unico tensor de dimension (len(tensor_list), 1, HEIGHT, WIDTH)
        stacked_tensor = torch.stack(tensor_list)

        self._logger.debug('Final dataset shape: {}'.format(stacked_tensor.shape))
        print("Final dataset shape: " + str(stacked_tensor.shape))

        self.data = stacked_tensor
        self.labels = labels_list

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

    @property
    def num_classes(self):
        return 7
