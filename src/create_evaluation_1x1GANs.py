import string
import re
import json
import pandas as pd
from datetime import datetime
import numpy as np
import math
import random
from collections import OrderedDict
from datetime import date

import os
import glob
import sys
from shutil import copyfile
from scipy.stats import shapiro

root_path = '.'
os.listdir(root_path)


def get_independent_run_params(path):
    parameters = None
    for filename in glob.iglob(path + '/*/*.log'):
        for line in open(filename, 'r'):
            if 'Parameters: ' in line:
                splitted_data = re.split("Parameters: ", line)
                parameters = json.loads(str(splitted_data[1]).replace("\'", "\"").replace("True", "true").replace("False", "false").replace("None", "null"))
                break
    return parameters


def get_iterations(parameters):
    return parameters['trainer']['n_iterations'] if not (parameters is None) else parameters


def get_result(gan_path):
    log_file = os.path.dirname(gan_path) + '/output.log'
    for line in open(log_file, 'r'):
        if 'yielded a score of' in line:
            splitted_data = re.split(",|\)|\(", line)
            fid = float(splitted_data[-3])
            tvd = float(splitted_data[-2])
            break
    return fid, tvd

def create_evaluation_run(gan_path, config_file, i):
    output_dir = os.path.dirname(gan_path)
    dev_number = 0 if random.random() < 0.5 else 1
    back_ground = '' if i%15==0 else '&'
    print(' CUDA_VISIBLE_DEVICES={}  python main.py evaluate --generator {}  -f {}  -o {}  2> {}/output.log {}'.format(
        dev_number, gan_path, config_file, output_dir, output_dir, back_ground))


def get_subfolders(root):
    return [dI for dI in os.listdir(root) if os.path.isdir(os.path.join(root, dI))]


def path_1x1_gan_training_results(path):
    paths = []
    for filepath in glob.iglob(path + '/*/gen*.pkl'):
            paths.append(filepath)
    return paths


master_path = '/media/toutouh/224001034000DF81/Documents/gan_1x1_bookchapter/lipizzaner-gan/src/output/lipizzaner_gan/master/'
distributed_path = '/media/toutouh/224001034000DF81/Documents/gan_1x1_bookchapter/lipizzaner-gan/src/output/lipizzaner_gan/distributed/mnist/'

independent_runs = get_subfolders(master_path)

gans_1x1 = []
for independent_run in independent_runs:
    new_1x1gan_path =  path_1x1_gan_training_results(master_path + independent_run)
    if len(new_1x1gan_path) == 1:
        parameters = get_independent_run_params(distributed_path + independent_run)
        iterations = get_iterations(parameters)
        if iterations >= 200:
            gans_1x1 = gans_1x1 + new_1x1gan_path

print(gans_1x1[0])
print('Creating {} runst to evaluate GANSs (1x1).'.format(len(gans_1x1)))
config_file  = 'configuration/quickstart-weights-optimization/mnist.yml'


#for i, gan in enumerate(gans_1x1):
#    create_evaluation_run(gan, config_file, i)

scores = {}
scores['fid'] = []
scores['tvd'] = []

for i, gan in enumerate(gans_1x1):
    fid, tvd = get_result(gan)
    scores['fid'].append(fid)
    scores['tvd'].append(tvd)

results = pd.DataFrame(scores)
results.to_csv('fid-tvd-results.csv', index=False)
results = results[results['fid']<40]


import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

corr, _ = spearmanr(results['fid'], results['tvd'])
print('Spearmans correlation: %.3f' % corr)

corr, _ = pearsonr(results['fid'], results['tvd'])
print('Pearsons correlation: %.3f' % corr)


fig = plt.figure()
ax = plt.axes()
plt.scatter(results['fid'], results['tvd'])
ax.tick_params(direction='out')
ax.set_ylabel(r"TVD")
ax.set_xlabel(r"FID")
fig.tight_layout()
fig.savefig("fid-tvd-100.png", dpi=300)
fig.show()

