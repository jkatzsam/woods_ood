#$1 must belong to:
#  woods, woods_nn, energy, energy_vos, OE
#
#$2 must belong to:
#  cifar10, cifar100
#
#$3 must belong to:
#  svhn, lsun_c, lsun_r,
#  isun, dtd, places, tinyimages_300k
#
#$4 must belong to:
#  svhn, lsun_c, lsun_r,
#  isun, dtd, places, tinyimages_300k

learning_rate=0.001
batch_size=128
ngpu=1
prefetch=4
epochs=100
script=train.py
gpu=0

load_pretrained='snapshots/pretrained'
checkpoints_dir=''
results_dir="results"

pi=0.1

score="$1"
dataset="$2"
aux_out_dataset="$3"
test_out_dataset="$4"

echo "running $score with dataset $dataset, aux_out_dataset $aux_out_dataset test_out_dataset $test_out_dataset, pi=$pi on GPU $gpu"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu CUDA_LAUNCH_BLOCKING=1 python $script $dataset \
--model wrn --score "$1" --learning_rate $learning_rate --epochs $epochs \
--test_out_dataset "$4" --aux_out_dataset "$3" --batch_size=$batch_size --ngpu=$ngpu \
--prefetch=$prefetch  --results_dir=$results_dir \
--checkpoints_dir=$checkpoints_dir --load_pretrained=$load_pretrained