#!/bin/bash


prefix="/home/cbalint/work/DNN/deepnet/deepnet/examples/ortho/"
datastats="/usr/lib64/python2.7/site-packages/deepnet/compute_data_stats.py"

echo Computing mean / variance
python ${datastats} ${prefix}/ortho/ortho.pbtxt ${prefix}/ortho/main_ortho.npz train_data || exit 1
python ${datastats} ${prefix}/ortho/ortho.pbtxt ${prefix}/ortho/test_ortho.npz test_data || exit 1
#python ${datastats} ${prefix}/ortho/ortho.pbtxt ${prefix}/ortho/unlabelled_ortho.npz unlabelled_data || exit 1

rm -rf save/test
rm -rf save/train
rm -rf save/validation

echo "RBM 1"
deep-trainer.py deep_rbm_layer1.pbtxt train-deep-layer1.pbtxt eval.pbtxt || exit 1
python /usr/lib64/python2.7/site-packages/deepnet/extract_neural_net_representation.py save/deep_dbn_1layer_LAST train-deep-layer1.pbtxt save/ hidden1

echo "RBM 2"
deep-trainer.py deep_rbm_layer2.pbtxt train-deep-layer2.pbtxt eval.pbtxt || exit 1
python /usr/lib64/python2.7/site-packages/deepnet/extract_neural_net_representation.py save/deep_rbm_2layer_LAST train-deep-layer2.pbtxt save/ hidden2

echo "RBM 3"
rm -rf save/train/hidden1*
rm -rf save/validation/hidden1*
deep-trainer.py deep_rbm_layer3.pbtxt train-deep-layer3.pbtxt eval.pbtxt || exit 1
python /usr/lib64/python2.7/site-packages/deepnet/extract_neural_net_representation.py save/deep_rbm_3layer_LAST train-deep-layer3.pbtxt save/ hidden3

echo "RBM 4"
rm -rf save/train/hidden2*
rm -rf save/validation/hidden2*
deep-trainer.py deep_rbm_layer4.pbtxt train-deep-layer4.pbtxt eval.pbtxt || exit 1
python /usr/lib64/python2.7/site-packages/deepnet/extract_neural_net_representation.py save/deep_rbm_4layer_LAST train-deep-layer4.pbtxt save/ hidden4

echo "RBM 5"
rm -rf save/train/hidden3*
rm -rf save/validation/hidden3*
deep-trainer.py deep_rbm_layer5.pbtxt train-deep-layer5.pbtxt eval.pbtxt || exit 1
python /usr/lib64/python2.7/site-packages/deepnet/extract_neural_net_representation.py save/deep_rbm_5layer_LAST train-deep-layer5.pbtxt save/ hidden5

echo "RBM 6"
rm -rf save/train/hidden4*
rm -rf save/validation/hidden4*
deep-trainer.py deep_rbm_layer6.pbtxt train-deep-layer6.pbtxt eval.pbtxt || exit 1
python /usr/lib64/python2.7/site-packages/deepnet/extract_neural_net_representation.py save/deep_rbm_6layer_LAST train-deep-layer6.pbtxt save/ hidden6

echo "RBM 7"
rm -rf save/train/hidden5*
rm -rf save/validation/hidden5*
deep-trainer.py deep_rbm_layer7.pbtxt train-deep-layer7.pbtxt eval.pbtxt || exit 1
python /usr/lib64/python2.7/site-packages/deepnet/extract_neural_net_representation.py save/deep_rbm_7layer_LAST train-deep-layer7.pbtxt save/ hidden7

echo "RBM 8"
rm -rf save/train/hidden6*
rm -rf save/validation/hidden6*
deep-trainer.py deep_rbm_layer8.pbtxt train-deep-layer8.pbtxt eval.pbtxt || exit 1
python /usr/lib64/python2.7/site-packages/deepnet/extract_neural_net_representation.py save/deep_rbm_8layer_LAST train-deep-layer8.pbtxt save/ hidden8

echo "RBM 9"
rm -rf save/train/hidden7*
rm -rf save/validation/hidden7*
deep-trainer.py deep_rbm_layer9.pbtxt train-deep-layer9.pbtxt eval.pbtxt || exit 1
python /usr/lib64/python2.7/site-packages/deepnet/extract_neural_net_representation.py save/deep_rbm_9layer_LAST train-deep-layer9.pbtxt save/ hidden9

deep-trainer.py deep_autoencode.pbtxt train-autoencode.pbtxt eval.pbtxt

python /usr/lib64/python2.7/site-packages/deepnet/extract_neural_net_representation.py \
       save/ortho_autoencoder_LAST train-autoencode.pbtxt save/ hidden9


./extract-test.py save/train/hidden9-00001-of-00001.npy dump.ply

