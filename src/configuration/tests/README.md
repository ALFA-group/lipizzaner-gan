### Configuration files to test new Lipizzaner functionalities

- **checkpointing**: It is a new feature to store the current status of each client node (cell) in its output folder. The stored information includes the *genome* (the network), the current iteration, learning rate, and all the information needed to resume the experiment from this given checkpoint.
- **weights-optimization**: It is used to test the optimization of the mixture weights at the enf of the training process. It basically call this function without performing any training epoch.
- **data-dieting**: It allows the selection of the portion of the training dataset to be used in each cell to train the networks. The samples of this reduced dataset are randomly picked.
- **mustangs**: Applyong Mustangs, the idea of randomlu picking a loss function from three different ones (BCE, MSE, and heuristic losses). It is introduced in **Spatial evolutionary generative adversarial networks**


Jamal Toutouh, Erik Hemberg, and Una-May O'Reilly. 2019. Spatial evolutionary generative adversarial networks. In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '19), Manuel López-Ibáñez (Ed.). ACM, New York, NY, USA, 472-480. DOI: https://doi.org/10.1145/3321707.3321860
