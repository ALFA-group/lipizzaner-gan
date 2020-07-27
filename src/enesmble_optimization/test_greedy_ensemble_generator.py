"""
Author: Jamal Toutouh (toutouh@mit.edu) - www.jamal.es

This code is part of the research of our paper "Re-purposing Heterogeneous Generative Ensembles with Evolutionary
Computation" presented during GECCO 2020 (https://doi.org/10.1145/3377930.3390229)

test_greedy_ensemble_generator.py contains the code to test the greedy approaches created to define ensembles of (GAN)
 generators by using evolutionary computing. It uses GreedyEnsembleGenerator class to address the problem.
"""

import pathlib
import sys

sys.path.insert(
    0, str(pathlib.Path(__file__).parent.parent.absolute())
)  # To change the folder path to use Lipizzaner files
from greedy_for_ensemble_generator import GreedyEnsembleGenerator

if len(sys.argv) < 4:
    print("Error: More parameters required")
    print(
        "Usage: python test_greedy_ensemble_generator.py <greedy mode> <ensemble size> "
        "<max iterations without improvements> [<output file>]"
    )
    print("\t <greedy mode>: iterative or random")
else:
    mode = sys.argv[1]
    ensemble_max_size = int(sys.argv[2])
    max_time_without_improvements = int(sys.argv[3])
    output_file = sys.argv[4] if len(sys.argv) == 5 else ""

    dataset = "mnist"
    precision = 10
    greedy = GreedyEnsembleGenerator(
        dataset,
        ensemble_max_size,
        precision,
        max_time_without_improvements,
        generators_prefix="mnist-generator",
        generators_path="./mnist-generators/",
        mode=mode,
        output_file=output_file,
    )

    greedy.create_ensemble()
