import os

from helpers.configuration_container import ConfigurationContainer
from helpers.ignore_label_dataset import IgnoreLabelDataset
from torchvision import transforms
from training.mixture.combined_score import CombinedCalculator
from training.mixture.constant_score import ConstantCalculator
from training.mixture.fid_score import FIDCalculator
from training.mixture.gaussian_score import (GaussianToyDistancesCalculator1D,
                                             GaussianToyDistancesCalculator2D)
from training.mixture.inception_score import InceptionCalculator
from training.mixture.prdc_score import PRDCCalculator, ToRGB


class ScoreCalculatorFactory:
    @staticmethod
    def create():
        cc = ConfigurationContainer.instance()
        settings = cc.settings["trainer"]["params"]

        if "score" not in settings:
            return None

        score_type = settings["score"].get("type", None)
        dataloader = cc.create_instance(cc.settings["dataloader"]["dataset_name"])
        # Downloads dataset if its not yet available
        loaded = dataloader.load()

        if score_type == "gaussian_toy_distances_2d":
            number_of_modes = cc.settings["dataloader"]["number_of_modes"]
            return GaussianToyDistancesCalculator2D(dataloader.dataset.points(number_of_modes))
        elif score_type == "gaussian_toy_distances_1d":
            return GaussianToyDistancesCalculator1D(next(iter(loaded)))
        elif score_type == "fid":
            transforms_op = [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
            dataset_name = cc.settings["dataloader"]["dataset_name"]
            if dataset_name == "mnist":
                if cc.settings["network"]["name"] == "ssgan_convolutional_mnist":
                    transforms_op = [transforms.Resize([64, 64])] + transforms_op
            elif dataset_name != "mnist_fashion":
                # Need to reshape for RGB dataset as required by pre-trained InceptionV3
                transforms_op = [transforms.Resize([64, 64])] + transforms_op

            split_param_keyword, split_param_value = ("split", "train") if dataset_name == "svhn" else ("train", True)
            dataset_params = {
                "root": os.path.join(cc.settings["general"]["output_dir"], "data"),
                split_param_keyword: split_param_value,
                "transform": transforms.Compose(transforms_op),
            }
            dataset = dataloader.dataset(**dataset_params)

            return FIDCalculator(
                IgnoreLabelDataset(dataset),
                cuda=cc.settings["master"].get("cuda", False),
                n_samples=settings["score"].get("score_sample_size", 10000),
            )
        elif score_type == "inception_score":
            # CUDA may not work when multiple  nodes, as it uses high amounts of GPU memory (~3GB per instance)
            return InceptionCalculator(cuda=cc.settings["master"].get("cuda", False), resize=True)
        elif score_type == "constant":
            return ConstantCalculator(cuda=cc.settings["master"].get("cuda", False), resize=True)
        elif score_type == "prdc":
            transforms_op = [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
            dataset_params = {
                "root": os.path.join(cc.settings["general"]["output_dir"], "data"),
                "train": True,
                "transform": transforms.Compose(transforms_op),
            }
            dataset = dataloader.dataset(**dataset_params)

            return PRDCCalculator(
                IgnoreLabelDataset(dataset),
                cuda=cc.settings["master"].get("cuda", False),
                n_samples=settings["score"].get("score_sample_size", 10000),
                nearest_k=settings["score"].get("nearest_k", 5),
            )
        elif score_type == "fid&prdc":
            use_random_vgg = settings["score"].get("use_random_vgg", False)
            prdc_transforms_op = None
            fid_transforms_op = [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
            dataset_name = cc.settings["dataloader"]["dataset_name"]
            if use_random_vgg:
                prdc_transforms_op = [ToRGB(), transforms.Resize(224)] + fid_transforms_op
            elif dataset_name == "mnist":
                if cc.settings["network"]["name"] == "ssgan_convolutional_mnist":
                    fid_transforms_op = [transforms.Resize([64, 64])] + fid_transforms_op
            elif dataset_name != "mnist_fashion":
                # Need to reshape for RGB dataset as required by pre-trained InceptionV3
                fid_transforms_op = [transforms.Resize([64, 64])] + fid_transforms_op

            fid_dataset_params = {
                "root": os.path.join(cc.settings["general"]["output_dir"], "data"),
                "train": True,
                "transform": transforms.Compose(fid_transforms_op),
            }
            fid_dataset = dataloader.dataset(**fid_dataset_params)

            if prdc_transforms_op is None:
                prdc_dataset = fid_dataset
            else:
                prdc_dataset_params = {
                    "root": os.path.join(cc.settings["general"]["output_dir"], "data"),
                    "train": True,
                    "transform": transforms.Compose(prdc_transforms_op),
                }
                prdc_dataset = dataloader.dataset(**prdc_dataset_params)

            return CombinedCalculator(
                prdc_dataset=IgnoreLabelDataset(prdc_dataset),
                fid_dataset=IgnoreLabelDataset(fid_dataset),
                cuda=settings["score"].get("cuda", False),
                n_samples=settings["score"].get("score_sample_size", 10000),
                nearest_k=settings["score"].get("nearest_k", 5),
                use_random_vgg=use_random_vgg,
            )
        else:
            raise Exception(
                'Mixture score type {} is not supported. Use either "inception_score" or "fid".'.format(score_type)
            )
