import losswise
import numpy as np

from helpers.configuration_container import ConfigurationContainer


class CoevLosswiseLogger():
    def __init__(self, method_name):
        self.method_name = method_name

        self.cc = ConfigurationContainer.instance()

        self.step_sizes_dis = []
        self.step_sizes_gen = []
        self.graphs = {}
        self.session = None
        self.w_gen_previous = None
        self.w_dis_previous = None

    def init_session(self, n_iterations, population_gen, population_dis):
        if not self.cc.is_losswise_enabled:
            return

        self.session = losswise.Session(tag=self.method_name, max_iter=n_iterations, params=self.cc.settings)
        self.graphs['loss'] = self.session.graph('loss')
        self.graphs['step_size'] = self.session.graph('step_size')

        self.w_gen_previous = population_gen.individuals[0].genome.parameters
        self.w_dis_previous = population_dis.individuals[0].genome.parameters

    def log_best_individuals(self, current_iteration, population_gen, population_dis):
        if not self.cc.is_losswise_enabled:
            return

        self.graphs['loss'].append(current_iteration, {
            'L(g(x)) - {}'.format(self.method_name): float(population_gen.individuals[0].fitness),
            'L(d(x)) - {}'.format(self.method_name): float(population_dis.individuals[0].fitness)})
        self.graphs['step_size'].append(current_iteration, {
            'avg_step_size(g(x)) - {}'.format(self.method_name): np.mean(self.step_sizes_gen),
            'avg_step_size(d(x)) - {}'.format(self.method_name): np.mean(self.step_sizes_dis)})

    def append_stepsizes(self, population_gen, population_dis):
        if not self.cc.is_losswise_enabled:
            return

        self.step_sizes_gen.append(
            np.linalg.norm(population_gen.individuals[0].genome.parameters - self.w_gen_previous))
        self.step_sizes_dis.append(
            np.linalg.norm(population_dis.individuals[0].genome.parameters - self.w_dis_previous))

        self.w_gen_previous = population_gen.individuals[0].genome.parameters
        self.w_dis_previous = population_dis.individuals[0].genome.parameters

    def end_session(self):
        if not self.cc.is_losswise_enabled:
            return
        self.session.done()
