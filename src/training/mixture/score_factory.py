import os

from torchvision import transforms

from helpers.configuration_container import ConfigurationContainer
from helpers.ignore_label_dataset import IgnoreLabelDataset
from training.mixture.fid_score import FIDCalculator
from training.mixture.inception_score import InceptionCalculator
from training.mixture.constant_score import ConstantCalculator


class ScoreCalculatorFactory:

    @staticmethod
    def create():
        cc = ConfigurationContainer.instance()
        settings = cc.settings['trainer']['params']

        if 'score' not in settings:
            return None

        score_type = settings['score'].get('type', None)
        dataloader = cc.create_instance(cc.settings['dataloader']['dataset_name'])
        # Downloads dataset if its not yet available
        dataloader.load()

        if score_type == 'fid':
            dataset = dataloader.dataset(root=os.path.join(cc.settings['general']['output_dir'], 'data'), train=True,
                                         transform=transforms.Compose(
                                             [transforms.Resize([64, 64]), transforms.ToTensor(),
                                              transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]))

            return FIDCalculator(IgnoreLabelDataset(dataset), cuda=settings['score'].get('cuda', False),
                                 n_samples=settings['score'].get('score_sample_size', 10000))
        elif score_type == 'inception_score':
            # CUDA may not work when multiple  nodes, as it uses high amounts of GPU memory (~3GB per instance)
            return InceptionCalculator(cuda=settings['score'].get('cuda', False), resize=True)
        elif score_type == 'constant':
            return ConstantCalculator(cuda=settings['score'].get('cuda', False), resize=True)
        else:
            raise Exception('Mixture score type {} is not supported. Use either "inception_score" or "fid".'
                            .format(score_type))
