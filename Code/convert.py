import glob
import os
import argparse
import routenet
import random
import db
import sklearn.model_selection


def get_lambda(delay):
    f = fname.split('/')[-1]
    df = f.split('_')[2]
    return df


def get_routing(routing):
    f = fname.split('/')[-1]
    df = f.split('_')[3:]
    return '_'.join(df)


parser = argparse.ArgumentParser(description='Convert to TF records')
parser.add_argument("--folder", help="Target Folder", required=True)
parser.add_argument("--db", help="Data Base", default=False, required=True)
parser.add_argument('--new', dest='isNew', action='store_true')
parser.add_argument('--old', dest='isNew', action='store_false')
parser.set_defaults(isNew=True)

args = parser.parse_args()

target = args.folder

database = db.DataSet(args.db)
database.connect()
database.create_table()

nodes = int(target.split('_')[0])

dataset_x, y = database.select(nodes)

if dataset_x == []:
    data = []
    for fname in glob.glob('datasets/' + target + '/delays/*.txt'):
        print(fname)
        tfname = fname.replace('txt', 'tfrecords')
        lam = get_lambda(fname)

        # Change the infer routing for the appropiate metod depending on the
        # topology

        routing = routenet.infer_routing_nsf4(
            fname) if target == '50_nodes' else routenet.infer_routing_nsf3(fname)
        routenet.make_tfrecord2(tfname,
                                'datasets/' + target + '/Network_' + target + '.ned',
                                routing,
                                fname,
                                args.isNew
                                )
        data.append((get_routing(routing), tfname, lam, nodes))

    database.insert_rows(data)
    dataset_x, y = database.select(nodes)

tfrecords = dataset_x
X_train, y_train = sklearn.model_selection.train_test_split(tfrecords, test_size=0.2, stratify=y)

for file in X_train:
    file_name = file.split('/')[-1]
    os.rename(file, 'datasets/' + target + '/tfrecords/train/' + file_name)


for file in y_train:
    file_name = file.split('/')[-1]
    os.rename(file, 'datasets/' + target + '/tfrecords/evaluate/' + file_name)
