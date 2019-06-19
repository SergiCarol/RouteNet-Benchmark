#!/bin/bash
# Creates the setup for working with a new topology

DIR="$1_nodes"
mkdir datasets/$DIR
mkdir datasets/$DIR/delays
mkdir datasets/$DIR/routing
mkdir datasets/$DIR/tfrecords
mkdir datasets/$DIR/tfrecords/train
mkdir datasets/$DIR/tfrecords/evalutate

echo "Make sure to copy the delays files and the routing files into the folders"
