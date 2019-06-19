# NetSimRouting

This program is used to generate destination aware only routing combinations, the parameters that the program accepts are the following:

    - name: the name of the topology to use for the routing creation.
    - n: Number of routing schemes to create.
    - min_length: the value to add to the SP as the value of cutoff. The cutoff value is always the shortest path between source-destination plus the min_length value. The default value is 0 for min length.
    - difficulty: routing placement policy, accepts three options:
        -- normal: random placement
        -- easy: sparse placement
        -- hard: overlapping placement

The algorithm works by generating paths between each source combination, each path has a maximum length of the length of the shortest path between the source and destination + the min_length value.

An example of use would be the following:

```sh
$ python3 NetSimRouting.py --name synth50 -n 1 --difficulty normal
```

# convert

This program is used to convert the data to tfrecords. It also splits the data between train and evaluation on a ratio of 80% and 20%. The split is done using an stratified sampling, using the lambda parameter as the variable to stratify.

```sh
$ python3 convert.py --folder 14_nodes --db test.sql
```

# eager_eval

Used to create the plots for evaluation. You ussualy never call this alone, it is part of the run.sh script.

Everytime you wish to check a new model, you must manualy change the directory in which the model is found.

# cdf

Same as above.

# run.sh

Help script used to ease the pre processing, training, and evaluation of the model.

The script accepts the following parameters:

    - load: pre processes the data using the convert script. The following parameters are required
        -- folder: folder to find the data.
        -- isNew: set to true if the new data format is needed.

    - train: trains the model, The following parameters are required
        -- folder: folder to find the data.
        -- checkpoint: where to store/find the model.
```sh
./run.sh train 50_nodes_route
```

    - predict: predicts the metrics for a given evaluation samples
        -- folder: folder to find the data.
        -- nodes: number of nodes of the topology that is wished to predict.
        -- checkpoint: where to store/find the model.
```sh
./run.sh predict 14_nodes 14 50_nodes_route
```

    - plots: plots the evaluation data and it sotres it in the current_results folder. 