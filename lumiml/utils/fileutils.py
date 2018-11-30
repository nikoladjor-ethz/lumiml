import os
import time
import dill
import warnings


def save_to_file(result_df, filename, result_log_path=None):
    tnow_ = time.strftime('%y_%m_%d_%H_%M')
    # file name generated based on current time
    current_path = os.getcwd()
    # check existence of the folder where data should be stored
    if result_log_path is None:
        # create default result log and write warning about it
        warnings.warn('Path to save data not defined. The default name result_log with current time will be used.',
                      UserWarning)
        log_name = 'result_log' + '_' + tnow_
        os.mkdir(current_path + '/' + log_name)
        file_name_full = log_name + '/' + tnow_ + '_' + filename + '.txt'
    elif not os.path.isdir(result_log_path):
        # requested folder does not exists, write warning message and create the folder within current working folder
        os.mkdir(current_path+'/'+result_log_path)
        file_name_full = result_log_path + tnow_ + '_' + filename + '.txt'
    else:
        # all good, create the file name based on the path
        file_name_full = result_log_path + tnow_ + '_' + filename + '.txt'

    with open(file_name_full, 'wb') as fp:
        dill.dump(result_df, fp)
        fp.close()

    return file_name_full


def load_from_file(path_to_file):
    with open(path_to_file, 'rb') as fp:
        data = dill.load(fp)
        fp.close()
    return data.copy()
