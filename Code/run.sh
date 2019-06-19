#!/bin/bash

if [[ "$1" = "load" ]]; then

    python3 convert.py --folder $2 --isNew $3

fi

if [[ "$1" = "train" ]]; then

    python3 routenet.py --ini config.ini train --hparams="l2=0.1,dropout_rate=0.5,link_state_dim=32,path_state_dim=32,readout_units=256,learning_rate=0.001,T=8,node_count=14, predict_count=14"  --train  ./datasets/$2/tfrecords/train/*.tfrecords --train_steps 10 --eval_ ./datasets/$2/tfrecords/evaluate/*.tfrecords --epochs 500 --model_dir ./CheckPoints/$3 

fi

if [[ "$1" = "predict" ]]; then

    python3 routenet.py --ini config.ini predict --hparams="l2=0.1,dropout_rate=0.5,link_state_dim=16,path_state_dim=32,readout_units=256,learning_rate=0.001,T=8,node_count=14,predict_count=$3"  --predict ./datasets/$2/tfrecords/evaluate/*.tfrecords --model_dir ./CheckPoints/$4/
fi

if [[ "$1" = "train_multiple" ]]; then

    for i in {1..50..2}
        do
        python3 routenet.py --ini config.ini train --hparams="l2=0.1,dropout_rate=0.5,link_state_dim=16,path_state_dim=32,readout_units=256,learning_rate=0.001,T=8,node_count=$3, predict_count=$3, isNew=False"  --train  ./datasets/$2/tfrecords/train/*.tfrecords --train_steps 10 --eval_ ./datasets/$2/tfrecords/evaluate/*.tfrecords --epochs 5 --model_dir ./CheckPoints/train_multiple

        python3 routenet.py --ini config.ini train --hparams="l2=0.1,dropout_rate=0.5,link_state_dim=16,path_state_dim=32,readout_units=256,learning_rate=0.001,T=8,node_count=$5, predict_count=$5, isNew=True"  --train  ./datasets/$4/tfrecords/train/*.tfrecords --train_steps 10 --eval_ ./datasets/$4/tfrecords/evaluate/*.tfrecords --epochs 5 --model_dir ./CheckPoints/train_multiple
        
        done
fi

if [[ "$1" = "plots" ]]; then
    mkdir current_results
    # Results evaluated with 14 nodes
    python3 eager_eval.py --name dGlobal_0_8_AL_2_k_8 --routing Routing_AL_2_k_8 --nodes 14
    python3 eager_eval.py --name dGlobal_0_8_SP_k_0 --routing Routing_SP_k_0 --nodes 14
    python3 eager_eval.py --name dGlobal_0_10_AL_1_k_0 --routing Routing_AL_1_k_0 --nodes 14
    python3 eager_eval.py --name dGlobal_0_15_SP_k_58 --routing Routing_SP_k_58 --nodes 14

    # Results evaluated with 24 nodes
    python3 eager_eval.py --name dGlobal_G_0_12_W_2_k_7 --routing RoutingGeant2_W_2_k_7 --nodes 24
    python3 eager_eval.py --name dGlobal_G_0_8_AL_2_k_0 --routing RoutingGeant2_AL_2_k_0 --nodes 24
    python3 eager_eval.py --name dGlobal_G_0_8_SP_k_17 --routing RoutingGeant2_SP_k_17 --nodes 24
    python3 eager_eval.py --name dGlobal_G_0_12_SP_k_46 --routing RoutingGeant2_SP_k_46 --nodes 24  
    python3 eager_eval.py --name dGlobal_G_0_12_W_4_k_0 --routing RoutingGeant2_W_4_k_0 --nodes 24
    python3 eager_eval.py --name dGlobal_G_0_15_W_7_k_6 --routing RoutingGeant2_W_7_k_6 --nodes 24
    python3 eager_eval.py --name dGlobal_G_0_10_W_1_k_8 --routing RoutingGeant2_W_1_k_8 --nodes 24
    python3 eager_eval.py --name dGlobal_G_0_10_SP_k_11 --routing RoutingGeant2_SP_k_11 --nodes 24
    python3 eager_eval.py --name dGlobal_G_0_15_W_1_k_0 --routing RoutingGeant2_W_1_k_0 --nodes 24
    python3 eager_eval.py --name dGlobal_G_0_10_SP_k_38 --routing RoutingGeant2_SP_k_38 --nodes 24
    python3 eager_eval.py --name dGlobal_G_0_10_AL_2_k_6  --routing RoutingGeant2_AL_2_k_6 --nodes 24

    # Results evaluated with 50 nodes NEW
    python3 eager_eval.py --name results_synth50_9_SP_k_9 --routing SP_k_9 --nodes 50 --isNew
    python3 eager_eval.py --name results_synth50_9_SP_k_52 --routing SP_k_52 --nodes 50 --isNew
    python3 eager_eval.py --name results_synth50_9_SP_k_31 --routing SP_k_31 --nodes 50 --isNew
    python3 eager_eval.py --name results_synth50_9_W_10_k_2 --routing W_10_k_2 --nodes 50 --isNew
    python3 eager_eval.py --name results_synth50_12_SP_k_20 --routing SP_k_20 --nodes 50 --isNew
    python3 eager_eval.py --name results_synth50_12_SP_k_46 --routing SP_k_46 --nodes 50 --isNew
    python3 eager_eval.py --name results_synth50_12_SP_k_85 --routing SP_k_85 --nodes 50 --isNew
    python3 eager_eval.py --name results_synth50_12_W_5_k_7 --routing W_5_k_7 --nodes 50 --isNew
    python3 eager_eval.py --name results_synth50_15_SP_k_51 --routing SP_k_51 --nodes 50 --isNew
    python3 eager_eval.py --name results_synth50_15_SP_k_90 --routing SP_k_90 --nodes 50 --isNew
    python3 eager_eval.py --name results_synth50_15_SP_k_13 --routing SP_k_13 --nodes 50 --isNew
    python3 eager_eval.py --name results_synth50_15_W_8_k_1 --routing W_8_k_1 --nodes 50 --isNew
    
    python3 cdf.py --file results.npz
fi
