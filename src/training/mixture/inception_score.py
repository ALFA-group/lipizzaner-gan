import numpy as np
import torch
import torch.utils.data
from scipy.stats import entropy
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.models.inception import inception_v3

from training.mixture.score_calculator import ScoreCalculator


class InceptionCalculator(ScoreCalculator):

    def __init__(self, cuda=False, batch_size=32, resize=True):
        """
        :param cuda: Whether or not to run on GPU. WARNING: Requires enormous amounts of memory.
        :param batch_size: Batch size for feeding into Inception v3
        :param resize: If set to true, the input images will be resized to the inception model's required size
        """

        self.dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.inception_model = inception_v3(pretrained=True, transform_input=False).type(self.dtype)
        self.inception_model.eval()
        self.batch_size = batch_size
        self.resize = resize

    def calculate(self, imgs, exact=True):
        """
        Calculate the inception score for the input image dataset
        :param exact: If set to true, the full dataset will be used (instead of only 3 splits).
        Results will be more accurate, but this takes more time.
        :param imgs: Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
        """

        length = len(imgs)
        splits = 10 if exact else 3

        assert self.batch_size > 0
        assert length > self.batch_size

        # Set up dataloader
        dataloader = torch.utils.data.DataLoader(imgs, batch_size=self.batch_size)
        up = nn.Upsample(size=(299, 299), mode='bilinear').type(self.dtype)

        def get_pred(x):
            if self.resize:
                x = up(x)
            x = self.inception_model(x)
            return F.softmax(x, dim=1).data.cpu().numpy()

        # Get predictions
        preds = np.zeros((length, 1000))

        for i, batch in enumerate(dataloader, 0):
            batch = batch.type(self.dtype)
            batchv = Variable(batch)
            batch_size_i = batch.size()[0]

            preds[i * self.batch_size:i * self.batch_size + batch_size_i] = get_pred(batchv)
            if i % 100 == 0:
                print('Batch {}/{}'.format(i, len(dataloader)))

        # Now compute the mean kl-div
        split_scores = []

        for k in range(splits):
            part = preds[k * (length // splits): (k + 1) * (length // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        return np.mean(split_scores), np.std(split_scores)

    @property
    def is_reversed(self):
        return False
