"""
Author: Jamal Toutouh (toutouh@mit.edu) - www.jamal.es

This code is part of the research of our paper "Re-purposing Heterogeneous Generative Ensembles with Evolutionary
Computation" presented during GECCO 2020 (https://doi.org/10.1145/3377930.3390229)

test_ga_ensemble_generator.py contains the code to test the evolutionary approaches created to define ensembles of (GAN)
 generators by using evolutionary computing. It uses GAEnsembleGenerator class to address the problem.
"""

import pathlib
import sys

sys.path.insert(
    0, str(pathlib.Path(__file__).parent.parent.absolute())
)  # To change the folder path to use Lipizzaner files
from ga_for_ensemble_generator import GAEnsembleGenerator

if len(sys.argv) < 7:
    print("Error: More parameters required")
    print(
        "Usage: python test_ga_ensemble_generator.py <evolutionary approach> <fitness metric> <crossovver prob.> "
        "<mutation prob.> <number of fitness evaluations> <population size> [<output file>]"
    )
    print("\t <evolutionary approach>: reo-gen or nreo-gen")
    print("\t <fitness metric>: tvd or fid")
    sys.exit(-1)

evolutionary_approach = sys.argv[1]
fitness_metric = sys.argv[2]
crossover_probability = float(sys.argv[3])
mutation_probability = float(sys.argv[4])
number_of_fitness_evaluations = int(sys.argv[5])
population_size = int(sys.argv[6])
output_file = sys.argv[7] if len(sys.argv) == 8 else ""


generators_path = "./mnist-generators/"
generators_prefix = "mnist-generator"
number_of_generations = 0
dataset = "mnist"
show_info_iteration = 2
min_ensemble_size, max_ensemble_size = 5, 7
ga = GAEnsembleGenerator(
    dataset,
    min_ensemble_size,
    max_ensemble_size,
    generators_path,
    generators_prefix,
    fitness_metric,
    evolutionary_approach,
    population_size,
    number_of_generations,
    number_of_fitness_evaluations,
    mutation_probability,
    crossover_probability,
    show_info_iteration,
    output_file,
)


ga.evolutionary_loop()
