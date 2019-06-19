
# coding: utf-8
import numpy as np
import tensorflow as tf
import matplotlib

import matplotlib.pyplot as plt
import routenet as upc
import configparser
import argparse
import random


parser = argparse.ArgumentParser()
parser.add_argument("--name", help="Topology name", type=str)
parser.add_argument("--routing", help="", type=str)
parser.add_argument("--nodes", help="", type=str)
parser.add_argument('--isNew', action='store_true')

args = parser.parse_args()

print("Currently running", args.name, args.routing, args.nodes)

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
    "l2=0.01,dropout_rate=0.5,link_state_dim=32,path_state_dim=32,readout_units=256,learning_rate=0.001,T=8, node_count=14")

model = upc.ComnetModel(hparams)
model.build()

saver = tfe.Saver(model.variables)
# Change this to point to the model directory
saver.restore('CheckPoints/50AP/model.ckpt-42256')


R = upc.load_routing('datasets/' + args.nodes + '_nodes/routing/' + args.routing + '.txt')
con, n = upc.ned2lists('datasets/' + args.nodes + '_nodes/Network_' + args.nodes + '_nodes.ned')
paths = upc.make_paths(R, con)
link_indices, path_indices, sequ_indices = upc.make_indices(paths)

if args.isNew:
    Global, bw, delay_index, jitter_index, packets_index, drop_index = upc.load(
        'datasets/' + args.nodes + '_nodes/delays/' + args.name + '.txt', n, full=True, isNew=True)
else:
    Global, bw, delay_index= upc.load(
        'datasets/' + args.nodes + '_nodes/delays/' + args.name + '.txt', n, isNew=False)

delay = Global.take(delay_index, axis=1).values
TMs = Global.take(bw, axis=1).values

if args.isNew:
    packets = Global.take(packets_index, axis=1).values
    drops = Global.take(drop_index, axis=1).values

n_paths = delay.shape[1]
n_links = max(max(paths)) + 1
n_total = len(path_indices)
i = random.randint(1, len(Global) - 1)  # Grab just one traffic matrix


tm = tf.convert_to_tensor(
    (TMs[i, :] - traffic_mean) / traffic_std, dtype=tf.float32)

if args.isNew:
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

# set training=True to enable dropout, this approximates posterior predictive distribution
hats = [model(feature, training=True).numpy() for i in range(50)]

# scaling and shifting parameters from training set (S10)
hats = delay_std * np.concatenate(hats, axis=1) + delay_mean

prediction = np.median(hats, axis=1)

conf = np.percentile(hats, q=[5, 95], axis=1)
grid = plt.GridSpec(8, 8, hspace=0.25, wspace=0.2)

ax_main = plt.subplot(grid[:-1, :-1])
ax = ax_main
# 90%
xerr = [prediction - np.percentile(hats, q=5, axis=1),
        np.percentile(hats, q=95, axis=1) - prediction]

print("total amount of points", len(prediction))
ax.errorbar(x=prediction, y=delay[i, :], fmt='o', alpha=0.6)

m = max(delay[i, :])


ax_right = plt.subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
ax_bottom = plt.subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])

x = prediction
y = delay[i, :]

binwidth = 0.25
xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
lim = (int(xymax / binwidth) + 1) * binwidth

ax.set_xlim((0, lim))
ax.set_ylim((0, lim))


ax.plot([0, 1.2 * m], [0, 1.2 * m], 'k')
ax.grid(True)
ax.set_xlabel('prediction')
ax.set_ylabel('True delay')
ax.set_title(args.nodes + ' nodes ' + args.name)


ax_bottom.set_xlim(ax.get_xlim())
ax_right.set_ylim(ax.get_ylim())

ax_bottom.hist(prediction, density=True, bins=40)
ax_bottom.grid(True)
ax_right.hist(delay[i, :], density=True, orientation='horizontal', bins=40)
ax_right.grid(True)
plt.figure(figsize=(8, 8))
manager = plt.get_current_fig_manager()
#manager.resize(*manager.window.maxsize()) # Un comment this if working with an UI and
                                           # wish to save the figure

plt.show()

plt.savefig('current_results/' + args.nodes + '_' + args.name + '.png')
