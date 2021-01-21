from pathlib import Path
from matplotlib import pyplot
from scipy.stats import shapiro
import random
import imageio
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib.figure import figaspect
import seaborn as sns
sns.set(style="whitegrid")
sns.set(font_scale=2)
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
import seaborn as sns
import statistics

data_folder = '../../data/'
images_folder = '../../images/'


def get_stats_to_paint(df):
    df['max'] = df[list(df)].max(axis=1)
    df['min'] = df[list(df)].min(axis=1)
    df['mean'] = df[list(df)].mean(axis=1)
    df['median'] = df[list(df)].median(axis=1)

    return df

def show_evolution_of_df(data_df, specific_client_id=None, filename=None):

    data_df = get_stats_to_paint(data_df)
    print('columns')
    print(data_df.columns)

    if specific_client_id is not None:
        specific_client_data = data_df['{}'.format(specific_client_id)].values

    sns.set(style="whitegrid")
    sns.set_style("ticks")
    x = np.arange(data_df.shape[0])
    fig = plt.figure()
    # plt.margins(0)
    ax = plt.axes()
    ax.tick_params(direction='out', labelsize=12)
    ax.plot(x, data_df['median'].values, 'b-')
    ax.fill_between(x, data_df['min'].values, data_df['max'].values, color='b', alpha=0.3)
    if specific_client_id is not None:
        ax.plot(x, specific_client_data, 'r--')
    
    # columns = data_df.columns
    # resurrected_df_keys = []
    # for key in columns:
    #     print(key)
    #     if '-1_grid' in key:
    #         resurrected_df_keys.append(key)
    # # resurrected_df_keys = [key for key in columns if '-1_grid' in key]
    # for key in resurrected_df_keys:
    #     print("resurrected key is {}".format(key))
    #     ax.plot(x, data_df['{}'.format(key)].values(), 'r--')

    # no plot without labels on the axis
    ax.set_xlabel(r"Training epoch", fontweight='bold', fontsize=14)
    ax.set_ylabel(r"FID score", fontweight='bold', fontsize=14)

    # always call tight_layout before saving ;)
    fig.tight_layout()
    plt.show()
    if filename is not None:
        fig.savefig(images_folder + filename)

def show_accuracy_label_evolution(data_df, image_path, epoch):
    labels = list(range(10))
    sns.set(style="whitegrid")
    sns.set_style("ticks")
    fig, ax = plt.subplots()
    plt.ylim(0, 100)
    plt.bar(labels, data_df.iloc[epoch])
    plt.title('MNIST - Training epoch: {}'.format(epoch))
    plt.ylabel('Classification accuracy (%)')
    plt.xticks(labels)
    plt.savefig(image_path)
    #plt.show()


def show_all_evolution(data_label_acc, data_acc, data_fid, image_path, epoch, max_generations):

    epoch = (max_generations-1) if epoch >= max_generations else epoch

    labels = list(range(10))
    sns.set(style="whitegrid")
    sns.set_style("ticks")
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    #fig = plt.subplot(1, 2, 1)
    axes[0].set_ylim(0, 100)
    axes[0].bar(labels, data_label_acc.iloc[epoch])
    #axes[0].title('MNIST - Training epoch: {}'.format(epoch))
    axes[0].set_ylabel('Labels classification accuracy (%)')
    axes[0].set_xlabel('MNIST labels')
    axes[0].set_xticks(labels)

    data_acc = get_stats_to_paint(data_acc)
    data_fid = get_stats_to_paint(data_fid)
    sns.set(style="whitegrid")
    sns.set_style("ticks")
    x = np.arange(0, epoch + 1)
    #fig, ax1 = plt.subplot(1, 2, 2)
    color = 'tab:red'
    axes[1].set_xlabel('Training epoch')
    axes[1].set_ylabel('FID score', color=color)
    axes[1].set_ylim(0, 200)
    axes[1].plot(x, data_fid['min'][:epoch + 1], color=color)
    axes[1].tick_params(axis='y', labelcolor=color)
    ax2 = axes[1].twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Classification accuracy (%)', color=color)  # we already handled the x-label with ax1
    ax2.set_ylim(0, 100)
    ax2.plot(x, data_acc['max'][:epoch + 1], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.xlim(0, max_generations)
    plt.savefig(image_path)


def createa_video(acc_label_log_file, acc_log_file, fid_log_file, client_id, step=1):
    data_acc = pd.read_csv(acc_log_file, index_col=False)
    data_fid = pd.read_csv(fid_log_file, index_col=False)
    data_label_acc = pd.read_csv(acc_label_log_file, index_col=False)
    experiment = acc_label_log_file.split('/')[-1][:-4]
    data = dict()
    for label in range(10):
        col_name = '{} - {}'.format(client_id, label)
        data['{}'.format(label)] = data_label_acc[col_name].tolist()
    data_label_acc = pd.DataFrame(data)

    tmp = '/tmp/'
    images = []
    Path(images_folder + tmp).mkdir(parents=True, exist_ok=True)

    max_generations = data_label_acc.shape[0]
    for epoch in range(max_generations):
        if epoch % step == 0:
            image_path = images_folder+ tmp + experiment + '-{:04d}'.format(epoch) + '.png'
            #show_accuracy_label_evolution(data_df, image_path, i)
            #show_evolution_of_2df(data_acc, data_fid, image_path, epoch)
            show_all_evolution(data_label_acc, data_acc, data_fid, image_path, epoch, max_generations)
            print('Created frame {}'.format(epoch))
            images.append(imageio.imread(image_path))

    # Create some frames to stop at the ende
    for epoch in range(30):
        if epoch % step == 0:
            image_path = images_folder + tmp + experiment + '-{:04d}'.format(epoch) + '.png'
            # show_accuracy_label_evolution(data_df, image_path, i)
            # show_evolution_of_2df(data_acc, data_fid, image_path, epoch)
            show_all_evolution(data_label_acc, data_acc, data_fid, image_path, max_generations + epoch, max_generations)
            print('Created frame {}'.format(epoch))
            images.append(imageio.imread(image_path))
    imageio.mimsave(images_folder + '/' + experiment + '.gif', images)
    print('Finished: Created animation in file {}'.format(experiment + '.gif'))


def show_all_experiments_evolution(pattern, experiment='mnist', metric='fid', grid_size=None, filename=None, group=None, resurrected_client=None):
    print('Control that you are using the same experiment files')
    import glob
    grid_size_pattern = '{}_grid*'.format(grid_size) if grid_size is not None else '*'
    if group == None:
        files = [filepath for filepath in glob.iglob(data_folder + '/final/'+ experiment + '-' + metric + '*_' + pattern + '_*' + grid_size_pattern +'.csv')]
    else:
        files = [filepath for filepath in glob.iglob(data_folder + '/final/'+ group + '/' + experiment + '-' + metric + '*_' + pattern + '_*' + grid_size_pattern +'.csv')]
    print(len(files))
    print(files)
    df_list =  list()
    best_values_iteration_75 = [] # create list for 75, 100, 200 
    best_values_iteration_100 = []
    best_values_iteration_200 = []
    for file in files:
        df_list.append(pd.read_csv(file, index_col=False))
        other_df = pd.read_csv(file, index_col=False)
        best_values_iteration_75.append(other_df.iloc[75].min()) 
        best_values_iteration_100.append(other_df.iloc[100].min()) 
        best_values_iteration_200.append(other_df.iloc[-1].min()) 

    all_data = pd.concat(df_list, ignore_index=True, axis=1)
    final = all_data.iloc[-1] # last row
    first = all_data.iloc[0] # first row 
    iteration_75 = all_data.iloc[74]
    iteration_100 = all_data.iloc[99]

    print('--------------------')
    print('--------------------')
    print(all_data.min().mean())
    print(all_data.min().min())
    print(all_data.mean().mean())
    # print(final.std())
    print('--------------------')
    print(first.min())
    print(first.max())
    print(first.mean())
    print(first.std())
    print('\n--- Iteration 75 ---')
    print('min ' + str(min(best_values_iteration_75)))
    print('max ' + str(max(best_values_iteration_75)))
    print('mean ' + str(statistics.mean(best_values_iteration_75)))
    print('median ' + str(statistics.median(best_values_iteration_75)))
    print('std ' + str(statistics.stdev(best_values_iteration_75)))
    print('\n--- Iteration 100 ---')
    print('min ' + str(min(best_values_iteration_100)))
    print('max ' + str(max(best_values_iteration_100)))
    print('mean ' + str(statistics.mean(best_values_iteration_100)))
    print('median ' + str(statistics.median(best_values_iteration_100)))
    print('std ' + str(statistics.stdev(best_values_iteration_100)))
    print('\n--- Iteration 200 ---')
    print('min ' + str(min(best_values_iteration_200)))
    print('max ' + str(max(best_values_iteration_200)))
    print('mean ' + str(statistics.mean(best_values_iteration_200)))
    print('median ' + str(statistics.median(best_values_iteration_200)))
    print('std ' + str(statistics.stdev(best_values_iteration_200)))

    show_evolution_of_df(all_data, specific_client_id=resurrected_client, filename=filename)





def show_evolution(log_file, specific_client=None, filename=None):
    data_df = pd.read_csv(log_file, index_col=False)
    show_evolution_of_df(data_df, specific_client, filename)

def show_evolution_of_2df(data_acc, data_fid,  image_path, epoch):
    data_acc = get_stats_to_paint(data_acc)
    data_fid = get_stats_to_paint(data_fid)
    sns.set(style="whitegrid")
    sns.set_style("ticks")
    x = np.arange(0, epoch+1)
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Training epoch')
    ax1.set_ylabel('FID score', color=color)
    ax1.set_ylim(0,200)
    ax1.plot(x, data_fid['min'][:epoch+1], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Classification accuracy (%)', color=color)  # we already handled the x-label with ax1
    ax2.set_ylim(0, 100)
    ax2.plot(x, data_acc['max'][:epoch+1], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.xlim(0,100)
    plt.savefig(image_path)


def show_evolution_fid_vs_acc(acc_log_file, fid_log_file):
    data_acc = pd.read_csv(acc_log_file, index_col=False)
    data_fid = pd.read_csv(fid_log_file, index_col=False)
    show_evolution_of_2df(data_acc, data_fid)


metrics = ['fid', 'gen_loss', 'disc_loss', 'gen_lr', 'disc_lr', 'per label accuracy', 'training_accuracy', 'fid_ensemble_evolution']

client_id = 1
acc_label_log_file = '/home/jamal/Documents/Research/sourcecode/evaluate-lipizzaneer-output/data/evolution/mnist-per_label_accuracy-evolution-lipizzaner_2020-05-17_08-21.csv'
acc_log_file = '/home/jamal/Documents/Research/sourcecode/evaluate-lipizzaneer-output/data/evolution/mnist-training_accuracy-evolution-lipizzaner_2020-05-17_08-21.csv'
fid_log_file = '/home/jamal/Documents/Research/sourcecode/evaluate-lipizzaneer-output/data/evolution/mnist-fid-evolution-lipizzaner_2020-05-17_08-21.csv'
fid_log_file = '/home/jamal/Documents/Research/sourcecode/evaluate-lipizzaneer-output/data/evolution/mnist-fid-evolution-lipizzaner_2020-10-05_22-51.csv'
log_file = '/home/jamal/Documents/Research/sourcecode/lipizzaner-output-analysis/data/final/mnist-fid-evolution-lipizzaner_2020-10-30_10-59-25_grid-.csv'
#createa_video(acc_label_log_file, acc_log_file, fid_log_file, client_id, step=1)
#show_evolution_fid_vs_acc(acc_log_file, fid_log_file)

# show_evolution('/home/ubuntu/lipizzaner-output-analysis/data/final/mnist-fid-evolution-lipizzaner_2020-12-07_16-38-3_grid-.csv', filename='mnist_test_100.png')
# show_evolution('/nobackup/users/umustafi/projects/lipizzaner-gan/lipizzaner-output-analysis/data/final/mnist-fid-evolution-lipizzaner_2020-12-16_10-21-4_grid-.csv', filename='client_dying.png')
show_all_experiments_evolution('*', experiment='mnist', metric='fid', grid_size=8, filename='9_experiment.png', group='9_experiment')
# show_all_experiments_evolution('*', experiment='mnist', metric='fid', filename='copy_experiment_1.png', group='copy_experiment_1')
# show_evolution(log_file, 0, 'test3.png')

# show_all_experiments_evolution('2020-10-06')
#show_all_experiments_evolution('*', experiment='mnist', metric='fid_ensemble_evolution', grid_size=16)


#createa_video(acc_label_log_file, client_id)

#show_evolution('/home/jamal/Documents/Research/sourcecode/evaluate-lipizzaneer-output/data/output/evolution/mnist-training_accuracy-evolution-lipizzaner_2020-05-16_14-46.csv')