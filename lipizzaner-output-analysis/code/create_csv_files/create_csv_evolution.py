from pathlib import Path
import string
import re
import json
import pandas as pd
from datetime import datetime
import numpy as np
import math


from collections import OrderedDict
from datetime import date

import os
import glob
import sys
from scipy.stats import shapiro

output_folder = '../../../../lipizzaner-gan/src/output/'
data_folder = '../../data/'
dataset = 'mnist'  #'circular' 

def get_all_master_log_files():
    return [filepath for filepath in glob.iglob(output_folder + 'log/*.log')]

def get_distributed_log_files_given_master_log(master_log_filename):
    return [filepath for filepath in glob.iglob(output_folder +'lipizzaner_gan/distributed/' + dataset + '/*/*/' + master_log_filename)]

def get_independent_run_params(file_name):
    parameters = None
    for line in open(file_name, 'r'):
        if 'Parameters: ' in line:
            splitted_data = re.split("Parameters: ", line)
            parameters = json.loads(str(splitted_data[1]).replace("\'", "\"").replace("True", "true").replace("False", "false").replace("None", "null"))
    return parameters


def get_loss_type(parameters):
    return parameters['network']['loss'] if not (parameters is None) else parameters

def get_client_id(parameters):
    return parameters['general']['distribution']['client_id'] if not (parameters is None) else parameters

def get_iterations(parameters):
    return parameters['trainer']['n_iterations'] if not (parameters is None) else parameters

def get_batch_size(parameters):
    return parameters['dataloader']['batch_size'] if not (parameters is None) else parameters

def get_label_rate(parameters):
    return parameters['dataloader']['label_rate'] if not (parameters is None) else parameters

def split_equal(data):
    container = data.split("=")
    return container[0], container[1]

def get_metric_value(analized_data, metric='fid'):
    if metric == 'fid':
        return float(split_equal(analized_data[5])[1]) #score
    elif metric == 'gen_loss':
        return float(split_equal(analized_data[1])[1])
    elif metric == 'disc_loss':
        return float(split_equal(analized_data[2])[1])
    elif metric == 'gen_lr':
        return float(split_equal(analized_data[3])[1])
    elif metric == 'disc_lr':
        return float(split_equal(analized_data[4])[1])
    elif metric == 'training_accuracy':
        print(analized_data)

def get_evolution_one_client(client_log, metric='fid'):
    data = []
    f = open(client_log, 'r')
    line = f.readline()
    # print(client_log)
    while line:
        if metric =='training_accuracy' and 'Label Prediction Accuracy' in line:
            splitted_data = re.split(" |,|%", line)
            data.append(float(splitted_data[-3]))
        elif not metric in ['per_label_accuracy', 'training_accuracy', 'gen_vs_disc_loss'] and 'Iteration=' in line:
            splitted_data = re.split("- |,|%", line)
            analized_data = splitted_data[3:9]
            data.append(get_metric_value(analized_data, metric))
        elif metric == 'per_label_accuracy' and \
                'Label, Number of Labeled Data points, Classification Rate for this label' in line:
            data_row = []
            label = 0
            line = f.readline()
            while label <= 9:
                data_row.append(float(line.split(',')[-1]))
                label += 1
                line = f.readline()
            data.append(data_row)
        elif metric == 'gen_vs_disc_loss' and 'Iteration=' in line:
            data_dict = dict()
            splitted_data = re.split("- |,|%", line)
            analized_data = splitted_data[3:9]
            data_dict['generator_loss'], data_dict['discriminator_loss'] = get_metric_value(analized_data, 'gen_loss'), get_metric_value(analized_data, 'disc_loss')
            data.append(data_dict)
        elif metric == 'fid' and 'Iteration=' in line:
            splitted_data = re.split("- |,|%", line)
            analized_data = splitted_data[3:9]
            data.append(get_metric_value(analized_data, 'fid'))
        line = f.readline()

    if len(data)>0:
        return data
    else:
        None

def get_max_iterations_of_clients_dataset(dataset):
    max = 0
    for data in dataset.values():
        if len(data) > max: max=len(data)
    return max


def complete_data_with_last_iterations(dataset, iterations): #We need all runs having the same number of iterations to be able to store everything in a dataframe/csv
    for client, metric_values in dataset.items():
        iterations_needed = iterations - len(metric_values)
        dataset[client] = metric_values + [metric_values[-1]] * iterations_needed
    return dataset

def get_evolution_distributed(master_log_filename, metric='fid'):
    distributed_log_files = get_distributed_log_files_given_master_log(master_log_filename)
    print(distributed_log_files)

    data_set = dict()
    n_iterations = get_iterations(get_independent_run_params(distributed_log_files[0]))
    print(n_iterations)

    if n_iterations is None: return

    for distributed_log_file in distributed_log_files:
        client_id = get_client_id(get_independent_run_params(distributed_log_file))
        if metric != 'per_label_accuracy':
            data = get_evolution_one_client(distributed_log_file, metric)
            if data is None:
                continue
            # if len(data) < 200:
            #     break
            # if not data is None: print(len(data))
            if (metric == 'gen_vs_disc_loss') and not data is None and len(data) == n_iterations:
                aux_df = pd.DataFrame(data)
                data_set['gen_loss-{}'.format(client_id)] = aux_df['generator_loss'].tolist()
                data_set['disc_loss-{}'.format(client_id)] = aux_df['discriminator_loss'].tolist()
            elif not data is None: # and len(data) >= n_iterations:
                data_set['{}'.format(client_id)] = data
        else:
            data = get_evolution_one_client(distributed_log_file, metric)
            if not data is None and len(data) >= n_iterations:
                data = np.array(data).T
                for i in range(len(data)):
                    dict_label = '{} - {}'.format(client_id, i)
                    data_set[dict_label] = data[i]




    if len(data_set) > 1:
        max_iterations_of_clients = get_max_iterations_of_clients_dataset(data_set)
        data_set = complete_data_with_last_iterations(data_set, max_iterations_of_clients)
        pd.DataFrame(data_set).to_csv(data_folder + '/final/' + dataset + '-' + metric + '-evolution-' +
                                  master_log_filename[:-4] + '-{}_grid-'.format(len(data_set)) + '.csv', index=False)
        print(data_folder + '/final/' + dataset + '-' + metric + '-evolution-' +
                                  master_log_filename[:-4] + '.csv')


def get_fid_weight_evolution(master_log_file):
    f = open(master_log_file, 'r')
    line = f.readline()
    scores = list()
    grid_size = 0
    improvement = -1

    while line:
        if 'Init score:' in line:
            splitted_data = re.split(" |:|\t", line)
            scores.append(float(splitted_data[11]))
        elif 'Score of new weights:' in line:
            splitted_data = re.split(" |:|\t", line)
            scores.append(float(splitted_data[21]))
        elif 'Successfully started experiment on http' in line:
            grid_size += 1
        elif 'Score after mixture weight optimzation:' in line:
            splitted_data = re.split(" |:|\t|\n", line)
            score_after = float(splitted_data[-2][:-2])
            score_before = float(splitted_data[-9])
            improvement = score_before - score_after
            scores.append(float(score_after))
        line = f.readline()

    master_log_filename = master_log_file.split('/')[-1]
    if scores is not None and len(scores)>0:
        pd.DataFrame(scores).to_csv(data_folder + '/final/' + dataset + '-fid_ensemble_evolution-evolution-' +
                                      master_log_filename[:-4] + '-{}_grid-'.format(grid_size) + '.csv',
                                      index=False)
        print(data_folder + '/final/' + dataset + '-fid_ensemble_evolution-evolution-evolution-' +
              master_log_filename[:-4] + '.csv')
    return improvement, grid_size, score_after

def get_evolution(metric='fid'):
    print('Getting evolution of {}'.format(metric))
    Path(data_folder + '/final/').mkdir(parents=True, exist_ok=True)
    data_set = []
    processed_independent_runs = 0
    grid_sizes = list()
    improvements = list()
    scores_after = list()
    for master_log in get_all_master_log_files():
        master_log_filename = master_log.split('/')[-1]
        if metric == 'fid_ensemble_evolution': # Artificial Life -> Evolution of weights
            improvement, gridsize, score_after = get_fid_weight_evolution(master_log)
            if improvement > 0:
                improvements.append(improvement)
                grid_sizes.append(gridsize)
                scores_after.append(score_after)
            processed_independent_runs += 1
        else:
            if True:
                print(master_log_filename)
                distributed_log_files = get_distributed_log_files_given_master_log(master_log_filename)
                print(distributed_log_files)
                if len(distributed_log_files) != 0:
                    get_evolution_distributed(master_log_filename, metric)
                    processed_independent_runs += 1

    print('Processed {} independent runs. '.format(processed_independent_runs))
    if len(improvements)>0:
        improvements_df = pd.DataFrame({'grid-size': grid_sizes, 'improvement': improvements, 'score-after':scores_after})
        improvements_df.to_csv('improvements-' + dataset + '.csv', index=None)
        print(improvements_df)



metrics = ['fid', 'gen_loss', 'disc_loss', 'gen_lr', 'disc_lr', 'per_label_accuracy', 'training_accuracy', 'gen_vs_disc_loss', 'fid_ensemble_evolution']

get_evolution('fid')

# if len(sys.argv)<2:
#     print('We need an argument for the metric to get')
#     print('Metrics: {}'.format(metrics))
# else:
#     print('Creating evolution files of: {}'.format(metrics[int(sys.argv[1])]))
#     get_evolution(metrics[int(sys.argv[1])])


