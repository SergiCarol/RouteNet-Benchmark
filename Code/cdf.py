
# coding: utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import routenet as upc
import configparser
import argparse
import random
import glob


def calculate_true_delay(R, ned, data, isNew):
    R = upc.load_routing(R)
    con, n = upc.ned2lists(ned)
    paths = upc.make_paths(R, con)
    link_indices, path_indices, sequ_indices = upc.make_indices(paths)

    if isNew:
        Global, bw, delay_index, jitter_index, packets_index, drop_index = upc.load(
            data, n, full=True, isNew=True)
    else:
        Global, bw, delay_index = upc.load(
            data, n, isNew=False)

    delay = Global.take(delay_index, axis=1).values
    TMs = Global.take(bw, axis=1).values

    if isNew:
        packets = Global.take(packets_index, axis=1).values
        drops = Global.take(drop_index, axis=1).values

    n_paths = delay.shape[1]
    n_links = max(max(paths)) + 1
    n_total = len(path_indices)

    i = random.randint(0, len(Global) - 1)
    tm = tf.convert_to_tensor(
        (TMs[i, :] - traffic_mean) / traffic_std, dtype=tf.float32)

    if isNew:
        feature = {
            'traffic': tf.convert_to_tensor(tm, dtype=tf.float32),
            'links': link_indices,
            'paths': path_indices,
            'sequances': sequ_indices,
            'n_links': n_links,
            'n_paths': n_paths,
            'n_total': n_total,
            'drops': drops,
            'packets': packets
        }
    else:
        feature = {
            'traffic': tf.convert_to_tensor(tm, dtype=tf.float32),
            'links': link_indices,
            'paths': path_indices,
            'sequances': sequ_indices,
            'n_links': n_links,
            'n_paths': n_paths,
            'n_total': n_total
        }
    return (delay[i, :], feature)


def calculate_delay_from_model(feature):

    hats = [model(feature, training=True).numpy() for i in range(50)]

    hats = delay_std * np.concatenate(hats, axis=1) + delay_mean

    prediction = np.median(hats, axis=1)

    return prediction


def r_squared(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    r_squared = 1 - (sum((y - (slope * x + intercept))**2) /
                     ((len(y) - 1) * np.var(y, ddof=1)))
    return r_squared


parser = argparse.ArgumentParser()
parser.add_argument("--file", help="Read result file and calculate from there",
                    type=str, required=False)

args = parser.parse_args()
try:
    res = np.load(args.file)
except:
    res = None
res = None
if res is not None:
    mre_14 = res['mre_14']
    mre_24 = res['mre_24']
    mre_50 = res['mre_50']
    r_2_14 = res['r_2_14']
    r_2_24 = res['r_2_24']
    r_2_50 = res['r_2_50']

    mse_14 = res['mse_14']
    mse_24 = res['mse_24']
    mse_50 = res['mse_50']
    p_corr_14 = res['p_corr_14']
    p_corr_24 = res['p_corr_24']
    p_corr_50 = res['p_corr_50']

    print('*' * 50)
    print('Results:')
    print('R^2')
    print("14 Nodes", r_2_14)
    print("24 Nodes", r_2_24)
    print("50 Nodes", r_2_50)
    print('*' * 50)
    print('*' * 50)
    print('MRE')
    print("14 Nodes", abs(np.average(mre_14)))
    print("24 Nodes", abs(np.average(mre_24)))
    print("50 Nodes", abs(np.average(mre_50)))
    print('*' * 50)
    print('MSE')
    print("14 Nodes", abs(np.average(mse_14)))
    print("24 Nodes", abs(np.average(mse_24)))
    print("50 Nodes", abs(np.average(mse_50)))
    print('*' * 50)
    print('P Correlation')
    print("14 Nodes", abs(np.average(p_corr_14)))
    print("24 Nodes", abs(np.average(p_corr_24)))
    print("50 Nodes", abs(np.average(p_corr_50)))
    print('*' * 50)
    plt.hist(mre_14, cumulative=True, label='14 Nodes',
             histtype='step', bins=100, alpha=0.8, color='blue', density=True)
    plt.hist(mre_24, cumulative=True, label='24 Nodes',
             histtype='step', bins=100, alpha=0.8, color='red', density=True)
    plt.hist(mre_50, cumulative=True, label='50 Nodes',
             histtype='step', bins=100, alpha=0.8, color='green', density=True)
    plt.title("CDF")

    plt.legend(prop={'size': 10})
    plt.show()

    exit()


tfe = tf.contrib.eager
tf.enable_eager_execution()

config = configparser.ConfigParser()
config.optionxform = str  # Disable lowercase conversion
config.read('config.ini')
normalization = config['Normalization']
delay_mean = float(normalization.get('mean_delay', 2.8))
traffic_mean = float(normalization.get('mean_traffic', 0.5))
delay_std = float(normalization.get('std_delay', 2.5))
traffic_std = float(normalization.get('std_traffic', .5))


hparams = upc.hparams.parse(
    "l2=0.1,dropout_rate=0.5,link_state_dim=32,path_state_dim=32,readout_units=256,learning_rate=0.001,T=8, node_count=50")

model = upc.ComnetModel(hparams)
model.build()

saver = tfe.Saver(model.variables)
saver.restore('CheckPoints/cdf_test/model.ckpt-19966')

# 14 Nodes

evaluate_records = random.sample(
    glob.glob('datasets/14_nodes/tfrecords/evaluate/*.tfrecords'), 50)
delays = []
path = 'datasets/14_nodes/delays/%s.txt'
for record in evaluate_records:
    tfrecord = record.split('/')[-1]
    file = tfrecord.split('.')[0]
    delays.append(path % file)


mre = []
r_2 = []
mse = []
mae = []
p_corr = []

for j in range(len(delays)):
    print("Using", upc.infer_routing_nsf3(delays[j]), delays[j])
    for i in range(30):
        y, feature = calculate_true_delay(upc.infer_routing_nsf3(delays[j]),
                                          'datasets/14_nodes/Network_14_nodes.ned',
                                          delays[j], False)

        x = calculate_delay_from_model(feature)

        relative_error = (y - x) / y
        square_error = (y - x)**2
        pearson = np.corrcoef(x, y)
        absolute_error = y - np.average(x)

        p_corr.append(pearson)
        mse.append(square_error)
        mre.append(relative_error)
        r_2.append(r_squared(x, y))
        mae.append(absolute_error)

mre = np.asarray(mre)
mse = np.asarray(mse)
p_corr = np.asarray(p_corr)
avg_mre_14 = np.average(mre, axis=0)
r_2_14 = np.average(r_2)
mse_14 = np.average(mse)
p_corr_14 = np.average(p_corr)
mae_14 = np.average(mae)

print("R Squared", r_2_14)

plt.hist(avg_mre_14, cumulative=True, label='14 Nodes',
         histtype='step', bins=100, alpha=0.8, color='blue', density=True)


# 24 Nodes
delays = random.sample(glob.glob('datasets/24_nodes/delays/*.txt'), 100)

mre = []
r_2 = []
mse = []
p_corr = []
mae = []

for j in range(len(delays)):
    print("Using", upc.infer_routing_geant(delays[j]), delays[j])
    for i in range(30):
        try:
            y, feature = calculate_true_delay(upc.infer_routing_geant(delays[j]),
                                              'datasets/24_nodes/Network_24_nodes.ned',
                                              delays[j], False)

            x = calculate_delay_from_model(feature)

            relative_error = (y - x) / y
            square_error = (y - x)**2
            pearson = np.corrcoef(x, y)
            absolute_error = y - np.average(x)

            p_corr.append(pearson)

            mse.append(square_error)
            mre.append(relative_error)
            r_2.append(r_squared(x, y))
            mae.append(absolute_error)

        except:
            continue

mre = np.asarray(mre)
p_corr = np.asarray(p_corr)
mse = np.asarray(mse)

avg_mre_24 = np.average(mre, axis=0)
r_2_24 = np.average(r_2)
mse_24 = np.average(mse)
p_corr_24 = np.average(p_corr)
mae_24 = np.average(mae)


print("R Squared", r_2_24)
plt.hist(avg_mre_24, cumulative=True, label='24 Nodes',
         histtype='step', bins=100, alpha=0.8, color='red', density=True)


# 50 Nodes

evaluate_records = random.sample(
    glob.glob('datasets/50_nodes/tfrecords/evaluate/*.tfrecords'), 100)
delays = []
path = 'datasets/50_nodes/delays/%s.txt'
for record in evaluate_records:
    tfrecord = record.split('/')[-1]
    file = tfrecord.split('.')[0]
    delays.append(path % file)


mre = []
r_2 = []
mse = []
p_corr = []
mae = []

for j in range(len(delays)):
    print("Using", upc.infer_routing_nsf4(delays[j]), delays[j])
    for i in range(30):

        try:
            y, feature = calculate_true_delay(upc.infer_routing_nsf4(delays[j]),
                                              'datasets/50_nodes/Network_50_nodes.ned',
                                              delays[j], True)

            x = calculate_delay_from_model(feature)

            relative_error = (y - x) / y
            square_error = (y - x)**2
            pearson = np.corrcoef(x, y)
            absolute_error = y - np.average(x)

            p_corr.append(pearson)
            mse.append(square_error)
            mre.append(relative_error)
            r_2.append(r_squared(x, y))
            mae.append(absolute_error)

        except:
            continue
mre = np.asarray(mre)
avg_mre_50 = np.average(mre, axis=0)
r_2_50 = np.average(r_2)
mse_50 = np.average(mse)
p_corr = np.asarray(p_corr)
mae_50 = np.average(mae)

p_corr_50 = np.average(p_corr)

print("R Squared", r_2_50)
plt.hist(avg_mre_50, cumulative=True, label='50 Nodes',
         histtype='step', bins=100, alpha=0.8, color='green', density=True)


np.savez(args.file, mre_14=avg_mre_14, mre_24=avg_mre_24, mre_50=avg_mre_50,
         r_2_14=r_2_14, r_2_24=r_2_24, r_2_50=r_2_50, mse_14=mse_14,
         mse_24=mse_24, mse_50=mse_50, p_corr_14=p_corr_14, p_corr_24=p_corr_24,
         p_corr_50=p_corr_50, mae_14=mae_14, mae_24=mae_24, mae_50=mae_50)

plt.title("CDF")

plt.legend(prop={'size': 10})
plt.show()
