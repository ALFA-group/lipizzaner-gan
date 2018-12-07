import csv
from collections import defaultdict
import matplotlib.pyplot as plt
from itertools import groupby, count
import pandas as pd
import numpy as np
import os

from sandbox.toy_problem.gaussian_gan import Individual, fct, NEGATIVE
from sandbox.toy_problem.opt_disc import f


def filter_best_per_generation(row):
    if 'individual_dis' in row:
        return row['individual_gen'] == '0' and row['individual_dis'] == '0'
    else:
        return row['individual_gen'] == '0'


def filter_best_of_last_generation(row):
    return row['individual_gen'] == '0' and row['generation'] == '99'


def filter_best_of_specific_generation(row, generation, run, quadrant):
    return row['individual_gen'] == '0' and row['generation'] == generation and row['round'] == run \
           and row['discr_collapse_type'] == '[{}, {}]'.format(quadrant[0], quadrant[1])


def line_plot(title, filename, target_file=None):
    reader = csv.DictReader(open(filename, newline=''))
    rows = list(filter(filter_best_per_generation, reader))

    values_to_group = [

        {'name': 'gen_adversarial_l1', 'label': 'left0', 'ls': 'dashed', 'c': '#FF0000'},
        {'name': 'gen_adversarial_l2', 'label': 'left1', 'ls': 'dashed', 'c': '#008000'},
        {'name': 'gen_adversarial_r1', 'label': 'right0', 'ls': 'dashed', 'c': '#FFA500'},
        {'name': 'gen_adversarial_r2', 'label': 'right1', 'ls': 'dashed', 'c': '#33FF33'},
        {'name': 'm1', 'label': 'µ1', 'ls': 'solid', 'c': '#00008B'},
        {'name': 'm2', 'label': 'µ2', 'ls': 'solid', 'c': '#ADD8E6'},
        {'name': 'm1_opt', 'label': 'µ1\'', 'ls': 'solid', 'c': '#000000'},
        {'name': 'm2_opt', 'label': 'µ2\'', 'ls': 'solid', 'c': '#000000'},
    ]
    df = pd.DataFrame(rows)

    def median(series, key):
        return series[key].astype(float).median()

    def mean(series, key):
        return series[key].astype(float).mean()

    def upper(series, key):
        return series[key].astype(float).max()

    def lower(series, key):
        return series[key].astype(float).min()

    results = []
    for k, v in df.groupby('generation'):
        val = {'generation': k}
        for grp in values_to_group:
            val[grp['name']] = mean(v, grp['name'])
            # if grp['name'] == "m1" or grp['name'] == 'm2':
            #     val[grp['name'] + '_upper'] = upper(v, grp['name'])
            #     val[grp['name'] + '_lower'] = lower(v, grp['name'])
        results.append(val)

    results = sorted(results, key=lambda key: int(key['generation']))

    x = [int(row['generation']) for row in results]
    for i, grp in enumerate(values_to_group):
        plt.plot(x, [float(row[grp['name']]) for row in results],
                 label=grp['label'], ls=grp['ls'], c=grp['c'])

        # if grp['name'] == "m1" or grp['name'] == 'm2':
        #     plt.fill_between(x, [float(row[grp['name'] + '_upper']) for row in results],
        #                      [float(row[grp['name'] + '_lower']) for row in results], alpha=0.3)

    plt.legend()
    plt.title(title)
    plt.xlim(xmin=0, xmax=100)
    if target_file is not None:
        plt.savefig(target_file, format='pdf')

    plt.show()


def discr_collapse_plot(filename, target_file=None):
    gen = Individual(0, [], 1)
    dis = Individual(0, [], 1)

    def close_enough(row):
        m1 = float(row['m1'])
        m2 = float(row['m2'])
        m1_opt = float(row['m1_opt'])
        m2_opt = float(row['m2_opt'])
        # Inverse cumsum
        z = np.asarray([m1, m2])
        z[1:] -= z[:-1].copy()
        gen.genome = z
        fitness = fct(gen, dis, m1_opt, m2_opt, 10, True)
        return abs(fitness - float(row['objective_best_fitness'])) < 0.1 or fitness < float(
            row['objective_best_fitness'])

    def map_coordinates(i, j):
        if i == 0 and j == 0:
            return -1, 1
        if i == 1 and j == 1:
            return 1, -1
        if i == 1 and j == 0:
            return 1, 1
        if i == 0 and j == 1:
            return -1, -1

    result = np.empty((2, 2))
    reader = csv.DictReader(open(filename, newline=''))
    rows = list(filter(filter_best_of_last_generation, reader))
    for i in range(2):
        for j in range(2):
            k, l = map_coordinates(i, j)
            curr_rows = [row for row in rows if row['discr_collapse_type'] == '[{}, {}]'.format(k, l)]
            if len(curr_rows) == 0:
                continue
            result[i][j] = sum(1 if close_enough(row) else 0 for row in curr_rows) / len(curr_rows)

    plt.xticks(range(2), ['negative', 'positive'])
    plt.yticks(range(2), ['positive', 'negative'])
    plt.imshow(np.array(result, dtype=np.float), cmap='jet', vmin=0, vmax=1)
    plt.colorbar()
    if target_file is not None:
        plt.savefig(target_file, format='pdf')
    plt.show()


def mode_collapse_plot(filename, target_file):
    def tvd(a, b):
        return sum(abs(a - b)) / 2

    def map_to_index(val):
        return int(round((val + 1) * 10))

    def close_enough(row):
        return tvd(np.asarray([float(row['m1']), float(row['m2'])]),
                   np.asarray([float(row['m1_opt']), float(row['m2_opt'])])) < 0.5

    gen = Individual(0, [], 1)
    dis = Individual(0, [], 1)

    def close_enough_to_opt(row):
        m1 = float(row['m1'])
        m2 = float(row['m2'])
        m1_opt = float(row['m1_opt'])
        m2_opt = float(row['m2_opt'])
        # Inverse cumsum
        z = np.asarray([m1, m2])
        z[1:] -= z[:-1].copy()
        gen.genome = z
        return fct(gen, dis, m1_opt, m2_opt, 10, True) < 0.1

    if not os.path.exists('mode_collapse_cache2.csv'):
        if os.path.exists('mode_collapse_cache.csv'):
            reader = csv.DictReader(open('mode_collapse_cache.csv', newline=''))
            rows = list(reader)
        else:
            reader = csv.DictReader(open(filename, newline=''))
            rows = list(filter(filter_best_of_last_generation, reader))

            with open('mode_collapse_cache.csv', 'w') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        result = np.empty((21, 21))
        result[:] = np.nan

        eps = 1e-4 * (1.0 - -1.0)
        num = int((1.0 - -1.0) / (0.1 - eps) + 1)

        k = 0
        for m1 in np.linspace(1.0, -1.0, num):
            for m2 in np.linspace(-1.0, 1.0 - k, num - int(round(k * 10))):
                curr_rows = [row for row in rows if float(row['m1_init']) == m1 and float(row['m2_init']) == m2]
                if len(curr_rows) == 0:
                    continue
                result[map_to_index(m1)][map_to_index(m2)] = sum(
                    1 if close_enough_to_opt(row) else 0 for row in curr_rows) / len(
                    curr_rows)

            k += 0.1
        np.savetxt('mode_collapse_cache2.csv', np.asarray(result), delimiter=',')
    else:
        result = np.loadtxt('mode_collapse_cache2.csv', delimiter=',')

    results2 = result[::2, ::2]
    plt.imshow(np.array(results2, dtype=np.float), cmap='jet', vmin=0, vmax=1)
    plt.xticks(range(11), np.around(np.linspace(-1.0, 1.0, 11), decimals=1))
    plt.yticks(range(11), np.around(np.linspace(-1.0, 1.0, 11), decimals=1))
    plt.setp(plt.axes().xaxis.get_ticklabels(), visible=False)
    plt.setp(plt.axes().yaxis.get_ticklabels(), visible=False)
    plt.setp(plt.axes().xaxis.get_ticklabels()[0::5], visible=True)
    plt.setp(plt.axes().yaxis.get_ticklabels()[0::5], visible=True)
    plt.title('Coevolutionary training')

    # plt.imshow(np.array(result, dtype=np.float), cmap='jet', vmin=0, vmax=1)
    # plt.xticks(range(21), np.around(np.linspace(-1.0, 1.0, 21), decimals=1))
    # plt.yticks(range(21), np.around(np.linspace(-1.0, 1.0, 21), decimals=1))
    # plt.setp(plt.axes().xaxis.get_ticklabels()[1::2], visible=False)
    # plt.setp(plt.axes().yaxis.get_ticklabels()[1::2], visible=False)
    plt.colorbar()
    plt.savefig(target_file, format='pdf')
    plt.show()


def discr_collapse_fitness_plots(title, generation, filename, target_file, quadrant):
    reader = csv.DictReader(open(filename, newline=''))
    row = list(filter(lambda r: filter_best_of_specific_generation(r, generation, '20', quadrant), reader))[0]

    params = {
        'p_mu_1': float(row['m1_opt']),
        'p_mu_2': float(row['m2_opt']),
        'q_mu_1': float(row['m1']),
        'q_mu_2': float(row['m2'])
    }

    l_1, r_1, l_2, r_2 = float(row['gen_adversarial_l1']), float(row['gen_adversarial_r1']), \
                         float(row['gen_adversarial_l2']), float(row['gen_adversarial_r2'])

    x = np.linspace(-10, 10, 500)
    f_x = f(x, params)

    disc_bounds = (l_1, r_1, l_2, r_2)
    f_bounds = f(disc_bounds, params)

    plt.plot(x, f_x, disc_bounds, f_bounds, 'r.')
    plt.fill_between(x, 0, f_x, where=f_x >= 0, facecolor='green', interpolate=True)

    x_1 = np.linspace(-10 if np.isinf(l_1) else l_1, 10 if np.isinf(r_1) else r_1, 10)
    plt.fill_between(x_1, -0.3, 0.3, facecolor='blue', alpha=0.5, label='$[left0,right0]$')

    x_2 = np.linspace(-10 if np.isinf(l_2) else l_2, 10 if np.isinf(r_2) else r_2, 10)
    plt.fill_between(x_2, -0.3, 0.3, facecolor='red', alpha=0.5, label='$[left1,right1]$')
    plt.xlabel('$x$')
    plt.ylabel('$p(x)-q(x)$')
    plt.title(title)
    plt.legend()
    if target_file is not None:
        plt.savefig(target_file, format='pdf')
    plt.show()


if __name__ == '__main__':

    result_dir = './experiment_results'

    discr_collapse_plot(filename=os.path.join(result_dir, 'discriminator_collapse.csv'),
                        target_file='plots/discr_collapse.pdf')
    mode_collapse_plot(filename=os.path.join(result_dir, 'mode_collapse.csv'),
                       target_file='plots/mode_collapse.pdf')
    line_plot(title='Coev Parallel (Symmetric)',
              filename=os.path.join(result_dir, 'parallel_symmetric.csv'),
              target_file='plots/coev_par_sym.pdf')
    line_plot(title='Coev Parallel (Asymmetric)',
              filename=os.path.join(result_dir, 'parallel_asymmetric.csv'),
              target_file='plots/coev_par_asym.pdf')
    line_plot(title='Coev Alternating (Symmetric), Discriminator collapse',
              filename=os.path.join(result_dir, 'discriminator_collapse.csv'),
              target_file='plots/coev_alt_sym_disc_coll.pdf')
    line_plot(title='Coev Alternating (Symmetric)',
              filename=os.path.join(result_dir, 'alternating_symmetric.csv'),
              target_file='plots/coev_alternating_sym.pdf')
    line_plot(title='Coev Alternating (Asymmetric)',
              filename=os.path.join(result_dir, 'alternating_asymmetric.csv'),
              target_file='plots/coev_alternating_asym.pdf')
    line_plot(title='Coev Alternating, optimal discriminator (Symmetric)',
              filename=os.path.join(result_dir, 'alternating_symmetric_opt_discr.csv'),
              target_file='plots/coev_alternating_sym_opt_discr.pdf')
    line_plot(title='Coev Alternating, optimal discriminator (Asymmetric)',
              filename=os.path.join(result_dir, 'alternating_asymmetric_opt_discr.csv'),
              target_file='plots/coev_alternating_asym_opt_discr.pdf')
    discr_collapse_fitness_plots(title='Discriminator collapse - Generation 1', generation='0',
                                 quadrant=[NEGATIVE, NEGATIVE],
                                 filename=os.path.join(result_dir, 'discriminator_collapse.csv'),
                                 target_file='plots/disc_collapse_gen1.pdf')
    discr_collapse_fitness_plots(title='Discriminator collapse - Generation 100', generation='99',
                                 quadrant=[NEGATIVE, NEGATIVE],
                                 filename=os.path.join(result_dir, 'discriminator_collapse.csv'),
                                 target_file='plots/disc_collapse_gen100.pdf')
