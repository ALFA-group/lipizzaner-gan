# Creating GAN Ensembles by using Evolutionary Computing

## Summary

Using ensmbles to improve generative adversarial networks (GANs) performance in generating accurate and diverese samples has shown a great success. Finding the best way to define these ensembles is not an easy task. 
This code applies two evolutionary algorithms (EAs) and two greedies to create ensembles of previously trained generators to maximize the diversity of the generated samples(i.e., improve TVD). 

These method have been presented in the paper [**Re-purposing Heterogeneous Generative Ensembles with Evolutionary Computation**](https://arxiv.org/abs/2003.13532), which has been accepted/published in **GECCO'20**. The information about the paper can be seen below.


## How-To

As these method are developed over **Lipizzaner**, it requires the installation of this software (see https://github.com/ALFA-group/lipizzaner-gan). 
In order to use our methods, the folder `enesmble_optimization/` should be copied inside the `src/` folder of Lipizzaner. 

Source code files in `enesmble_optimization/` folder:
- evolutionary_nonrestricted_ensemble_optimization.py: It implements the NREO-GEN method.
- evolutionary_restricted_ensemble_optimization.py: It implements the REO-GEN method.
- greedy_for_ensemble_generator.py: It implements both Greedy methods, iterative and random.

In order to test these methods, we provide test cases that use ten generators previously trained to generate MNIST samples.
- test_ga_ensemble_generator.py: It allows testing the EA methods (i.e., NREO-GEN and REO-GEN)
- test_greedy_ensemble_generator.py: It allows testing the Greedy methods.


## GECCO'20 Paper Information

#### Title: 
**Re-purposing Heterogeneous Generative Ensembles with Evolutionary Computation**

#### Abstract: 
Generative Adversarial Networks (GANs) are popular tools for generative modeling. The dynamics of their adversarial learning give rise to convergence pathologies during training such as mode and discriminator collapse. In machine learning, ensembles of predictors demonstrate better results than a single predictor for many tasks. In this study, we apply two evolutionary algorithms (EAs) to create ensembles to re-purpose generative models, i.e., given a set of heterogeneous generators that were optimized for one objective (e.g., minimize Frechet Inception Distance), create ensembles of them for optimizing a different objective (e.g., maximize the diversity of the generated samples). The first method is restricted by the exact size of the ensemble and the second method only restricts the upper bound of the ensemble size. Experimental analysis on the MNIST image benchmark demonstrates that both EA ensembles creation methods can re-purpose the models, without reducing their original functionality. The EA-based demonstrate significantly better performance compared to other heuristic-based methods. When comparing both evolutionary, the one with only an upper size bound on the ensemble size is the best.

#### ACM Reference Format:

Jamal Toutouh, Erik Hemberg, and Una-May O’Reilly. 2019. Re-purposing Heterogeneous Generative Ensembles with Evolutionary Computation. In *Genetic and Evolutionary Computation Conference (GECCO ’20), 2020.* ACM, New York, NY, USA, 9 pages. [https://doi.org/10.1145/3377930.3390229](https://arxiv.org/abs/2003.13532)

#### Bibtex Reference Format:

```
@inproceedings{Toutouh_GECO2020,
author = {Toutouh, Jamal and Hemberg, Erik and O’Reilly, Una-May},
title = {Re-purposing Heterogeneous Generative Ensembles with Evolutionary Computation},
year = {2020},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3377930.3390229},
doi = {10.1145/3377930.3390229},
booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference},
numpages = {9},
series = {GECCO ’2020}
}
```
