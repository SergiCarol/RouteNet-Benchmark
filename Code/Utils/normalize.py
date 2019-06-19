import numpy as np 
import os, random
import argparse
import tensorflow as tf
import configparser


def load_records(records, limit=30):
    files = []
    for folder in records:
        f = os.listdir(folder)
        sample = random.sample(f, limit)
        for file in sample:
            files.append(folder + '/' + file)
    return files


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Netwrok routing ML tool')
    parser.add_argument("--dir", help="Directories", nargs='+')
    parser.add_argument("--ini", help="Config file")

    args = parser.parse_args()
    print("Args dir", args.dir)
    records = load_records(args.dir)
    delays = []
    traffic = []
    for _ in range(5):
        for record in records:
            print(record)
            record_iterator = tf.python_io.tf_record_iterator(path=record)
            for string_record in record_iterator:
                example = tf.train.Example()
                example.ParseFromString(string_record)
                delays += example.features.feature['delay'].float_list.value
                traffic += example.features.feature['traffic'].float_list.value

    print('*' * 10)
    print('Delay')
    print('Mean', np.mean(delays)) 
    print('Std', np.std(delays))
    print('*' * 10)
    print('Traffic')
    print('Mean', np.mean(traffic)) 
    print('Std', np.std(traffic))
    
    config = configparser.ConfigParser()
    config.optionxform = str  # Disable lowercase conversion
    config.read(args.ini)
    config['Normalization']['mean_delay'] = str(np.mean(delays))
    config['Normalization']['std_delay'] = str(np.std(delays))
    config['Normalization']['mean_traffic'] = str(np.mean(traffic))
    config['Normalization']['std_traffic'] = str(np.std(traffic))
    with open('config.ini', 'w') as configfile:
        config.write(configfile)
