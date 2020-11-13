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

from matplotlib import pyplot
from scipy.stats import shapiro
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
import seaborn as sns
from nonparametric_tests import friedman_test

from scipy.stats import shapiro
from scipy.stats import ttest_ind
from scipy.stats import iqr
import scikit_posthocs as sp
from scipy.stats import shapiro
from random import sample, choices
from scipy.stats import iqr


root_path = '.'
root_path = './output-potimization/'
#root_path = './output-potimization/greddy-preliminars/'

def get_summary_files(prefix, method):
    # path_list = []
    # for filepath in glob.iglob(prefix + '/results-summary/1x1-GAN-data-partition-summary.csv'):
    #     print(filepath)
    #     path_list.append(filepath)
    # return path_list
    return [file for file in glob.iglob(prefix + '/screen-*{}*'.format(method))]

def parse_last_results_random_search(filepath):
    data = dict()
    f = open(filepath)
    for line in f:
        if 'TVD=' in line:
            strings = line.split('=')
            data['FIT-Min'] = float(strings[-1])
            data['TVD-Min'] = float(strings[-1])
        if 'FID=' in line:
            strings = line.split('=')
            data['FID'] = float(strings[-1])
        if 'Generators examined' in line and not 'Mixture:' in line:
            strings = line.split('=')
            data['Generators evaluated='] = int(strings[-1])
        if 'Execution time' in line:
            strings = line.split('=')
            data['Run time'] = float(strings[-1])
    return data

def parse_last_results_greddy_screen_file(filepath):
    data = dict()
    if 'random-search' in filepath:
        return data
    f = open(filepath)
    for line in f:
        if 'TVD=' in line:
            strings = line.split('=')
            data['FIT-Min'] = float(strings[-1])
            data['TVD-Min'] = float(strings[-1])
        if 'FID=' in line:
            strings = line.split('=')
            data['FID'] = float(strings[-1])
        if 'Generators examined' in line and not 'Mixture:' in line:
            strings = line.split('=')
            data['Generators evaluated='] = int(strings[-1])
        if 'Execution time' in line:
            strings = line.split('=')
            data['Run time'] = float(strings[-1])
    return data


def parse_last_results_ga_screen_file(filepath):
    data = dict()
    print('PARSING....')
    f = open(filepath)
    for line in f:
        if 'Number of generations' in line:
            strings = line.split('=')
            data['Number of generations'] = int(strings[-1])
        if 'Population size' in line:
            strings = line.split('=')
            data['Population size'] = int(strings[-1])
        if 'P_mu' in line:
            strings = line.split('=')
            data['P_mu'] = float(strings[-1])
        if 'P_cr' in line:
            strings = line.split('=')
            data['P_cr'] = float(strings[-1])
        if 'FIT' in line and 'Min' in line:
            strings = line.split(' ')
            data['FIT-Min'] = float(strings[-1])
        if 'FIT' in line and 'Max' in line:
            strings = line.split(' ')
            data['FIT-Max'] = float(strings[-1])
        if 'FIT' in line and 'Avg' in line:
            strings = line.split(' ')
            data['FIT-Avg'] = float(strings[-1])
        if 'FIT' in line and 'Std' in line:
            strings = line.split(' ')
            data['FIT-Std'] = float(strings[-1])
        if 'FID' in line and 'Min' in line:
            strings = line.split(' ')
            data['FID-Min'] = float(strings[-1])
        if 'FID' in line and 'Max' in line:
            strings = line.split(' ')
            data['FID-Max'] = float(strings[-1])
        if 'FID' in line and 'Avg' in line:
            strings = line.split(' ')
            data['FID-Avg'] = float(strings[-1])
        if 'FID' in line and 'Std' in line:
            strings = line.split(' ')
            data['FID-Std'] = float(strings[-1])
        if 'TVD' in line and 'Min' in line:
            strings = line.split(' ')
            data['TVD-Min'] = float(strings[-1])
        if 'TVD' in line and 'Max' in line:
            strings = line.split(' ')
            data['TVD-Max'] = float(strings[-1])
        if 'TVD' in line and 'Avg' in line:
            strings = line.split(' ')
            data['TVD-Avg'] = float(strings[-1])
        if 'TVD' in line and 'Std' in line:
            strings = line.split(' ')
            data['TVD-Std'] = float(strings[-1])
        if 'Generators examined' in line and not 'Mixture:' in line:
            strings = line.split('=')
            data['Generators evaluated='] = int(strings[-1])
        if 'Execution time' in line:
            strings = line.split('=')
            data['Run time'] = float(strings[-1])
    print(data)
    return data

def create_summary_dataframe_ga(algorithm):
    #log_files = get_summary_files(root_path, algorithm)
    data = [parse_last_results_ga_screen_file(log_file) for log_file in get_summary_files(root_path, algorithm)]
    data_df = pd.DataFrame(data)
    data_df = data_df.sort_values(by=['P_mu', 'P_cr'])
    data_df.to_csv(algorithm + '-results.csv', index=False)
    return data_df

def create_summary_dataframe_greddy(algorithm):
    #log_files = get_summary_files(root_path, algorithm)
    data = [parse_last_results_greddy_screen_file(log_file) for log_file in get_summary_files(root_path, algorithm)]
    data_df = pd.DataFrame(data).dropna()
    data_df.to_csv(algorithm + '-results.csv', index=False)
    return data_df

def create_summary_dataframe_random_search(algorithm):
    #log_files = get_summary_files(root_path, algorithm)
    data = [parse_last_results_random_search(log_file) for log_file in get_summary_files(root_path, algorithm)]
    data_df = pd.DataFrame(data).dropna()
    data_df.to_csv(algorithm + '-results.csv', index=False)
    return data_df

def get_basic_stats(data1):
    num = np.array(data1)
    iqr_range = iqr(num)
    mean_val = num.mean()
    median_val = np.median(num)
    max_val = num.max()
    min_val = num.min()
    count = len(data1)
    norm_std_val = num.std() / mean_val * 100
    std_val = num.std()

    stats = {}
    stats['mean'] = mean_val
    stats['norm_stdev'] = norm_std_val
    stats['stdev'] = std_val
    stats['min'] = min_val
    stats['median'] = median_val
    stats['max'] = max_val
    stats['iqr'] = iqr_range
    stats['count'] = count
    stats_string = '{:.3f}$\\pm${:.3f} & {:.3f} &  {:.3f}  &  {:.3f} & {:.3f}\\\ '.format(mean_val, std_val, min_val, median_val, iqr_range, max_val)
    #print(stats_string)
    #print(stats)
    return stats_string #mean_val, min_val


def evaluate_sensibility_analysis(data_df, metric):
    data_boxplot = dict()
    list_of_data = []
    for pm in [0.01, 0.05, 0.1]:
        for pcr in [0.25, 0.50, 0.75]:
            data = list(data_df.loc[(data_df['P_mu']==pm) & (data_df['P_cr']==pcr)][metric])
            list_of_data.append(data[0:5])
            stats = get_basic_stats(data)
            print('{} & {} & {}'.format(pm, pcr, stats))#print(aux_df)
            data_boxplot['{}-{}'.format(pm,pcr)] = data
    print(data_boxplot)

    if False:
        boxplot_df = pd.DataFrame(data_boxplot)
        w, h = figaspect(3 / 4)
        f, ax = plt.subplots(figsize=(w, h))
        # f, ax = plt.subplots()
        sns.set(style="whitegrid")
        ax.set_ylabel(metric, fontweight='bold')
        #ax.set_ylim(15, 80)
        ax = sns.boxplot(data=boxplot_df)
        #plt.savefig('figures/extended-selection-vs-no_selection-9-boxplot.png')
        plt.show()
    return list_of_data

def evaluate_results(data_df_list, metric):
    list_of_data = []
    data_boxplot = dict()
    for idx, data_df in zip(range(len(data_df_list)), data_df_list):
        data = list(data_df[metric])
        list_of_data.append(data)
        data_boxplot[str(idx)] = data
        stats = get_basic_stats(data)
        print('{} & {}'.format(idx, stats))
    print(data_boxplot)

    if False: #True:
        boxplot_df = pd.DataFrame(data_boxplot)
        w, h = figaspect(3 / 4)
        f, ax = plt.subplots(figsize=(w, h))
        # f, ax = plt.subplots()
        sns.set(style="whitegrid")
        ax.set_ylabel(metric, fontweight='bold')
        #ax.set_ylim(15, 80)
        ax = sns.boxplot(data=boxplot_df)
        #plt.savefig('figures/extended-selection-vs-no_selection-9-boxplot.png')
        plt.show()
    return list_of_data


# data = create_summary_dataframe('optimize-ga*just-weight-probabilty')
# evaluate_sensibility_analysis(data, 'FIT-Min')
# evaluate_sensibility_analysis(data, 'FIT-Max')
# evaluate_sensibility_analysis(data, 'FIT-Avg')
# evaluate_sensibility_analysis(data, 'FID-Min')
# evaluate_sensibility_analysis(data, 'Run time')

# data = create_summary_dataframe('optimize-ga*just-weight-probabilty')
# evaluate_sensibility_analysis(data, 'FIT-Min')
# data = create_summary_dataframe('optimize-ga*just-generator-index')
# evaluate_sensibility_analysis(data, 'FIT-Min')
# data = create_summary_dataframe('optimize-ga*generator-and-weight')
# evaluate_sensibility_analysis(data, 'FIT-Min')
data = create_summary_dataframe_ga('optimize-ga*combined-mutation')

list_data = evaluate_sensibility_analysis(data, 'FIT-Min')
print(list_data)
print(friedman_test(list_data[0], list_data[1], list_data[2], list_data[3], list_data[4], list_data[5], list_data[6], list_data[7]))



# data_list = []
# data = create_summary_dataframe_greddy('random')
# data_list.append(data)
# data = create_summary_dataframe_greddy('iterative')
# data_list.append(data)
# data = create_summary_dataframe_random_search('random-search')
# data_list.append(data)
# evaluate_results(data_list, 'FIT-Min')
# #print(data)

