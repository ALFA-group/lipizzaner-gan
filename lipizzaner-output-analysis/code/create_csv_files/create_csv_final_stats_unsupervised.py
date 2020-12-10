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
dataset = 'mnist' #'mnist'


def get_stats(values):
    num = np.array(values)
    minn = num.min()
    maxx = num.max()
    mean = num.mean()
    std = num.std()
    return minn, maxx, mean, std


def get_all_distributed_log_files():
    return [filepath for filepath in glob.iglob(output_folder + dataset + '/*/*/*.log')]


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


def get_grid_size(parameters):
    return len(parameters['general']['distribution']['client_nodes']) if not (parameters is None) else parameters

def get_label_rate(parameters):
    return parameters['dataloader']['label_rate'] if not (parameters is None) else parameters

def get_all_master_log_files():
    return [filepath for filepath in glob.iglob(output_folder + 'log/*.log')]


def split_best_result(line):
    splitted_data = re.split(' |\(|,|\)', line)
    return float(splitted_data[-4]), float(splitted_data[-2]), splitted_data[-7]


def get_fid_tvd_time_bestclient_from_master_log(master_log_path):
    fid, init_time = None, None
    for line in open(master_log_path, 'r'):
        splitted_data = re.split("- |,", line)
        if init_time is None:
            init_time = datetime.strptime(splitted_data[0], '%Y-%m-%d %H:%M:%S')
        if 'Stopping heartbeat...' in line:
            stop_time = datetime.strptime(splitted_data[0], '%Y-%m-%d %H:%M:%S')
        if 'Best result:' in line:
            fid, tvd, best_client = split_best_result(line)

    if not fid is None:
        execution_time = stop_time - init_time
        execution_time_minutes = execution_time.total_seconds() / 60
        return fid, tvd, execution_time_minutes, best_client, str(init_time) #datetime.strptime(init_time, '%Y-%m-%d %H:%M:%S')
    else:
        return None, None, None, None, None


def get_fid_tvd_time_results(get_accuracy=True):
    dataset = []
    processed_independent_runs = 0
    for master_log in get_all_master_log_files():
        data = dict()
        master_log_filename = master_log.split('/')[-1]
        distributed_log_files = get_distributed_log_files_given_master_log(master_log_filename)
        if len(distributed_log_files) != 0:
            independent_run_parameters = get_independent_run_params(distributed_log_files[0])
            n_iterations = get_iterations(independent_run_parameters)
            try:
                label_rate = get_label_rate(independent_run_parameters)
            except:
                label_rate = 1
            batch_size = get_batch_size(independent_run_parameters)
            grid_size = get_grid_size(independent_run_parameters)
            fid, tvd, execution_time_minutes, best_client, init_time = get_fid_tvd_time_bestclient_from_master_log(master_log)
            if not fid is None:
                data['init_time'] = init_time
                data['score'] = fid
                data['tvd'] = tvd
                data['execution_time'] = execution_time_minutes
                data['best FID client'] = best_client
                data['n_iterations'] = n_iterations
                data['grid_size'] = grid_size
                data['label_rate'] = label_rate
                data['batch_size'] = batch_size
                dataset.append(data)
                processed_independent_runs += 1

    print('Processed {} independent runs. '.format(processed_independent_runs))

    return pd.DataFrame(dataset)

def get_last_voting_stats_one_client(client_log, n_iterations):
    last_iteration = False
    subpop_accuracies = []
    most_voted = None
    for line in open(client_log, 'r'):
        if not last_iteration and 'training.ea.ea_trainer - Iteration' in line:
            splitted_data = re.split(" |,", line)
            last_iteration = (splitted_data[-4].isnumeric() and int(splitted_data[-4])==n_iterations)
        if last_iteration:
            splitted_data = re.split("/| |,", line)
            if ' - Test Accuracy: ' in line:
                subpop_accuracies.append(float(splitted_data[-3])/float(splitted_data[-2]))
            if ' - Majority Voting Test Accuracy: ' in line:
                most_voted = float(splitted_data[-3])/float(splitted_data[-2])
                break
    return most_voted, subpop_accuracies[1:6]

def get_last_voting_stats(master_log_filename, type='most voted'):
    distributed_log_files = get_distributed_log_files_given_master_log(master_log_filename)
    n_iterations = get_iterations(get_independent_run_params(distributed_log_files[0]))
    dataset = []
    for distributed_log_file in distributed_log_files:
        stats = dict()
        stats['acc stats client id'] = get_client_id(get_independent_run_params(distributed_log_file))
        stats['acc stats client folder'] = distributed_log_file.split('/')[-2]
        stats['most voted acc'], subpop_accuracy = get_last_voting_stats_one_client(distributed_log_file, n_iterations)
        if len(subpop_accuracy)>0:
            _, stats['max acc'], stats['mean acc'], stats['std acc'] = get_stats(subpop_accuracy)
            stats['improvement over max acc'] = stats['most voted acc'] - stats['max acc']
            stats['improvement over mean acc'] = stats['most voted acc'] - stats['mean acc']
            dataset.append(stats)

    data_df = pd.DataFrame(dataset)
    try:
        if type == 'most voted':
            return data_df.loc[data_df['most voted acc'].idxmax()]
        elif type == 'max':
            return data_df.loc[data_df['max acc'].idxmax()]
        elif type == 'imporvement over mean':
            return data_df.loc[data_df['improvement over mean acc'].idxmax()]
        elif type == 'improvement over max':
            return data_df.loc[data_df['improvement over max acc'].idxmax()]
    except:
        return 'No accuracy info'


#
# master_log_filename = 'lipizzaner_2020-05-16_14-46.log'
# get_last_voting_stats(master_log_filename)

#print(get_last_voting_stats('/home/jamal/Documents/Research/sourcecode/evaluate-lipizzaneer-output/data/output/lipizzaner_gan/distributed/mnist/2020-05-14_19-07-57/10612/lipizzaner_2020-05-14_19-07.log', 100))

results_df = get_fid_tvd_time_results(False)
results_df.to_csv(data_folder + dataset + '-summary_results-gaussian.csv', index=False)


