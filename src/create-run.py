import string
from datetime import datetime

from collections import OrderedDict
from datetime import date

import os
import glob
import sys
from scipy.stats import shapiro


def get_files(master_path):
    paths = []
    for filepath in glob.iglob(master_path):
        paths.append(filepath)
    return paths

path = 'output-selection/lipizzaner_gan/master/*/*'
paths = get_files(path)

print('source activate lipizzaner')

for i in range(len(paths)-1, 0, -1):
    print('killall python')
    print('sleep 2')
    code = 'python main.py optimize --mixture-source {} -o ./samples --sample-size {} -f {} > {}'.format(
        paths[i], 50000, 'configuration/data-partition/mnist-1.yml', paths[i] + '/weights-optimization.log')
    print(code)
    print('sleep 2')

sys.exit()

#python main.py optimize --mixture-source /media/toutouh/224001034000DF81/Documents/gan_1x1_bookchapter/lipizzaner-gan/src/output/lipizzaner_gan/master/2019-09-24_01-06-10/127.0.0.1-5008/ -o ./samples --sample-size 50000 -f configuration/data-partition/mnist-1.yml > salida.log


def split_equal(data):
    container = data.split("=")
    return container[0], container[1]


def get_datetime(str):
    return datetime.strptime(str, '%Y-%m-%d %H:%M:%S')


def get_runtime(data_frame, column): #in seconds
    first_last_time = data_frame.iloc[[0, -1]][column].values.tolist()
    return int((first_last_time[1] - first_last_time[0])/100000000)


def get_dataframe_concat_cols(list_df, column):
    list_df_cols = []
    list_df_names = []
    index = 0
    for df in list_df:
        list_df_cols.append(df[column])
        list_df_names.append('df_' + str(index))
        index += 1
    df_result = pd.concat(list_df_cols, axis=1, keys=list_df_names)
    return df_result

def get_iterations(list_df):
    list_iterations = []
    for df in list_df:
        list_iterations.append(int(df['iteration'].iloc[-1]))
    return list_iterations


def get_final_results(list_df):
    if len(list_df) == 0:
        return 0,0
    df_scores = get_dataframe_concat_cols(list_df, 'score')
    df_scores.fillna(1000)
    #df_times = get_dataframe_concat_cols(list_df, 'date_time')
    average_iterations = np.array(get_iterations(list_df)).mean()

    min_score = df_scores.min().min()
    #print(min_score)

    list_runtimes = []
    #for column in df_times:
    #    print(df_times[column])
    #    if not df_times[column] == 'NaN':
    #        df_times[column] = pd.to_datetime(df_times[column], format='%Y-%m-%d %H:%M:%S')
    #        list_runtimes.append(get_runtime(df_times, column))
    #average_runtime = np.array(list_runtimes).mean()

    return float(min_score), float(average_iterations)





def log_to_dataframe(filename):
    list_of_data = []
    data_storage = None
    batches_found = False
    num_batches = 0
    current_batch = -1

    for line in open(filename, 'r'):
        if 'Batch' in line and not batches_found:
            splitted_data = re.split(" |\/", line)
            current_batch = int(splitted_data[9])
            if current_batch < num_batches:
                batches_found = True
            else:
                num_batches = current_batch


        if 'Iteration=' in line:
            splitted_data = re.split("- |,", line)
            analized_data = splitted_data[3:9]
            data_storage = {}

            data_storage['iteration'] = int(split_equal(analized_data[0])[1])
            data_storage['f(Generator(x))'] = float(split_equal(analized_data[1])[1])
            data_storage['f(Discriminator(x))'] = float(split_equal(analized_data[2])[1])
            data_storage['lr_gen'] = float(split_equal(analized_data[3])[1])
            data_storage['lr_dis'] = float(split_equal(analized_data[4])[1])
            data_storage['score'] = float(split_equal(analized_data[5])[1])
            data_storage['date_time'] = str(get_datetime(splitted_data[0]))
            data_storage['num_batches'] = num_batches

        if data_storage is not None:
            list_of_data.append(data_storage)
            data_storage = None
    df = pd.DataFrame(list_of_data)
    if DESIRED_GRID_SIZE==1:        
        get_basic_stats_1x1_gan(df, filename)

    return df



def get_independent_runs_path(root_path, prefix):
    list_path = []
    for dirpath in glob.iglob(root_path + '/'+ prefix + '*/*/'):
        list_path.append(dirpath)
    return list_path


def get_independent_run_dataframes(independent_run_path):
    list_df = []
    for filepath in glob.iglob(independent_run_path + '*/*.log'):
        list_df.append(log_to_dataframe(filepath))
    return list_df

def get_independent_run_dataframes_egan(independent_run_path):
    list_df = []
    for filepath in glob.iglob(independent_run_path + '*/*/*.log'):
        data_df = log_to_dataframe_egan(filepath)
        list_df.append(data_df)
    return list_df


def get_stats(algorithm, array_metric):
    num = np.array(array_metric)
    best = num.min()
    worst = num.max()
    mean = num.mean()
    std = num.std()/mean*100
    stats_string = 'Variation: {}\t Independent runs: {}\t Mean: {}\t Std: {}\t Best: {}\t Worst: {}'.format(algorithm, len(array_metric), mean, std, best, worst)
    stats_string2 = '{}\t{}\t{}\t{}\t{}\t{}'.format(algorithm, len(array_metric), mean, std, best, worst)
    print(stats_string2)

def is_in_list(pattern, str_list):
    found = False
    for str in str_list:
        if pattern == str:
            found = True
            break
    return found



def get_subfolders(root):
    return [dI for dI in os.listdir(root) if os.path.isdir(os.path.join(root, dI))]


def get_just_log_folders(fodlders_list):
    cells = []
    for cell in fodlders_list:
        if 'log' in cell and 'lip-' in cell:
            cells.append(cell)
    return cells


def get_independent_runs_path(root, cells):
    independent_runs_list = []

    master_independent_runs = get_subfolders(root + '/' + cells[0] + '/')
    print(master_independent_runs)
    for master_independent_run in master_independent_runs:
        print()
        found_in_all = True
        for cell in cells:
            all_paths = get_subfolders(root + '/' + cell)
            if not is_in_list(master_independent_run, all_paths):
                found_in_all = False
                break
        if found_in_all:
            independent_runs_list.append(master_independent_run)
            print(master_independent_run)

    return independent_runs_list


def get_loss_type(parameters):
    return parameters['network']['loss'] if not (parameters is None) else parameters


def get_iterations(parameters):
    return parameters['trainer']['n_iterations'] if not (parameters is None) else parameters

def get_enabled_selection(parameters):
    return parameters['trainer']['params']['enable_selection'] if not (parameters is None) else parameters

def get_score_computing(parameters):
    return parameters['trainer']['params']['score']['enabled'] if not (parameters is None) else parameters


def get_independent_run_params(file_name):
    parameters = None
    for line in open(file_name, 'r'):
        if 'Parameters: ' in line:
            splitted_data = re.split("Parameters: ", line)
            parameters = json.loads(str(splitted_data[1]).replace("\'", "\"").replace("True", "true").replace("False", "false").replace("None", "null"))
    return parameters


#log_to_dataframe('./raw/lip-1-log/2019-04-05_13-43-55/6/lipizzaner_2019-04-05_13-43.log')
#sys.exit(0)

machine_index = 4
machine = ['lip-1', 'lip-3', 'lip-5', 'lip-6', 'desk-1']
root_path = '../data/raw/' + machine[machine_index] + '/output/lipizzaner_gan/distributed/mnist/'
log_root_path = '../data/raw/' + machine[machine_index] + '/output/log/'



root_path = '/home/jamaltoutouh/data-partition/lipizzaner-gan/src/output-selection/lipizzaner_gan/distributed/mnist/'
log_root_path = '/home/jamaltoutouh/data-partition/lipizzaner-gan/src/output-selection/log/'

root_path = '/media/toutouh/224001034000DF81/Documents/gan_1x1_bookchapter/lipizzaner-gan/src/output-selection/lipizzaner_gan/distributed/mnist/'
log_root_path = '/media/toutouh/224001034000DF81/Documents/gan_1x1_bookchapter/lipizzaner-gan/src/output-selection/log/'

root_path = '/media/toutouh/224001034000DF81/Documents/gan_1x1_bookchapter/lipizzaner-gan/src/output/lipizzaner_gan/distributed/mnist/'
log_root_path = '/media/toutouh/224001034000DF81/Documents/gan_1x1_bookchapter/lipizzaner-gan/src/output/log/'

independent_runs = get_subfolders(root_path)

dataframes_iterations = {}
dataframes_iterations_log_stats = {}


list_basic_stats_1x1_gan = []

def get_basic_stats_1x1_gan(df, filename):
    data_storage = {}

    if df.shape[0] > 199:
        init_time = get_datetime(df.iloc[0]['date_time'])
        end_time = get_datetime(df.iloc[-1]['date_time'])
        computation_time = (end_time - init_time).total_seconds() / 60

        data_storage['iterations'] = df.shape[0]
        data_storage['independent_run'] = filename 
        data_storage['score'] = df.iloc[-1]['score'].min()
        data_storage['computation_time'] = computation_time
        data_storage['num_chunks'] = int((df.iloc[-1]['num_batches']+1) / 66)     
        data_storage['num_batches'] = int(df.iloc[-1]['num_batches'] + 1)
        list_basic_stats_1x1_gan.append(data_storage)

def get_logs_stats(independent_run, num_batches, selection_enabled, score_enabled):
    filename = log_root_path + '/lipizzaner_' + independent_run + '.log'
    data_storage = {}
    score = 0

    init_time_read=False

    for line in open(filename, 'r'):
        splited_data = re.split("- |,", line)
        if not init_time_read:
            init_time = get_datetime(splited_data[0])
            init_time_read = True

        if 'Best result:' in line:
            splited_data = re.split("\(|,", line)
            score = float(splited_data[2])
            end_time = get_datetime(splited_data[0])
            break

    if score != 0:
        computation_time = (end_time - init_time).total_seconds() / 60
        data_storage['independent_run'] = independent_run
        data_storage['score'] = score
        data_storage['computation_time'] = computation_time
        data_storage['num_batches'] = num_batches
        data_storage['selection_enabled'] = selection_enabled
        data_storage['score_enabled'] = score_enabled
    return data_storage

print(independent_runs)


for independent_run in independent_runs:
    cells = get_subfolders(root_path + '/' + independent_run)
    #print('*********************************************')
    #print(cells)
    #print(len(cells))
    #print(independent_run)
    
    if len(cells) < 1:
        continue

    one_log_file = root_path + '/' + independent_run + '/' + cells[0] + '/lipizzaner_' + independent_run[:-3] + '.log'
    parameters = get_independent_run_params(one_log_file)
    iterations = get_iterations(parameters)
    selection_enabled = get_enabled_selection(parameters)
    score_enabled =  get_score_computing(parameters)


    

    if iterations is None:
        continue

    if not iterations in dataframes_iterations.keys():
        dataframes_iterations[iterations] = []
        dataframes_iterations_log_stats[iterations] = []

    path = root_path + '/' + independent_run + '/' + cells[0] + '/lipizzaner_' + independent_run[:-3] + '.log'
    dataframes_list = []

    
    if len(cells) == DESIRED_GRID_SIZE:

        for cell in cells:
            path = root_path + '/' + independent_run + '/' + cell + '/lipizzaner_' + independent_run[:-3] + '.log'
            #path=root_path + '/' + cell + '/' + independent_run + '/6/lipizzaner_' + independent_run[:-3] + '.log'
            df = log_to_dataframe(path).rename(index=str, columns={"score": cell})

            if df.empty:
                break
            if df.shape[0] < 200:
                break
            dataframes_list.append(df[[cell]])
            dataframes_iterations[iterations].append(df[[cell]])
            num_batches = max(df['num_batches']) + 1

    if len(cells) == DESIRED_GRID_SIZE and len(dataframes_list) == DESIRED_GRID_SIZE:
        all_cells_score = pd.concat(dataframes_list, axis=1)
        all_cells_score.to_csv('../data/processed/' + machine[machine_index] + '/' + str(iterations) + '-' + str(len(cells)) + '-score_' + independent_run + '.csv', index=False)
        dataframes_iterations_log_stats[iterations].append(get_logs_stats(independent_run[:-3], num_batches, selection_enabled, score_enabled))
        #dataframes_list.append(df[[cell]])
        dataframes_iterations[iterations].append(df[[cell]])

for key, value in dataframes_iterations.items():
    if len(value) > 0:
        pd.concat(value, axis=1).to_csv(str(key) + '-' + machine[machine_index] + '-' + str(DESIRED_GRID_SIZE) + '-all-results.csv', index=False)

if DESIRED_GRID_SIZE==1:
    dataframes_iterations_log_stats = list_basic_stats_1x1_gan

pd.DataFrame(dataframes_iterations_log_stats).to_csv('ALL-' + str(DESIRED_GRID_SIZE) + '-summary-results.csv', index=False)
sys.exit(0)


print(dataframes_iterations_log_stats)

for key, value in dataframes_iterations_log_stats.items():
    if len(value) > 0:
        print(key)
        pd.DataFrame(value).to_csv(str(key) + '-' + machine[machine_index] + '-' + str(DESIRED_GRID_SIZE) + '-summary-results.csv', index=False)
        print('SUMMARY:')
        print(pd.DataFrame(value))
        #pd.concat(value, axis=1).to_csv(str(key) + '-' + machine[machine_index] + '-summary-results.csv', index=False)


sys.exit(0)

