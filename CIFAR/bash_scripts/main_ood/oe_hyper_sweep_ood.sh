learning_rate=0.001
epochs=50
batch_size=128
ngpu=1
prefetch=4
script=train.py

load_pretrained='snapshots/pretrained'

test_out='tinyimages_300k'

for seed in 10; do
  for pi in 1 0.05 0.5 0.1 0.2; do
    for score in 'OE'; do
      for dataset in 'cifar100' 'cifar10'; do

        results_dir="results_ood_${seed}"
        checkpoints_dir="checkpoints_ood_${seed}"

        gpu=6
        oe_lambda=0.1
        echo "running $score with $dataset, $test_out, oe_lambda=$oe_lambda, pi=$pi on GPU $gpu"
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu CUDA_LAUNCH_BLOCKING=1 python $script $dataset \
        --model wrn --score $score --seed $seed --learning_rate $learning_rate --epochs $epochs \
        --test_out_dataset $test_out --aux_out_dataset $test_out --batch_size=$batch_size --ngpu=$ngpu \
        --prefetch=$prefetch --pi=$pi --oe_lambda=$oe_lambda --results_dir=$results_dir \
        --checkpoints_dir=$checkpoints_dir --load_pretrained=$load_pretrained &

        gpu=4
        oe_lambda=0.5
        echo "running $score with $dataset, $test_out, oe_lambda=$oe_lambda, pi=$pi on GPU $gpu"
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu CUDA_LAUNCH_BLOCKING=1 python $script $dataset \
        --model wrn --score $score --seed $seed --learning_rate $learning_rate --epochs $epochs \
        --test_out_dataset $test_out --aux_out_dataset $test_out --batch_size=$batch_size --ngpu=$ngpu \
        --prefetch=$prefetch --pi=$pi --oe_lambda=$oe_lambda --results_dir=$results_dir \
        --checkpoints_dir=$checkpoints_dir --load_pretrained=$load_pretrained &

        gpu=5
        oe_lambda=1
        echo "running $score with $dataset, $test_out, oe_lambda=$oe_lambda, pi=$pi on GPU $gpu"
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu CUDA_LAUNCH_BLOCKING=1 python $script $dataset \
        --model wrn --score $score --seed $seed --learning_rate $learning_rate --epochs $epochs \
        --test_out_dataset $test_out --aux_out_dataset $test_out --batch_size=$batch_size --ngpu=$ngpu \
        --prefetch=$prefetch --pi=$pi --oe_lambda=$oe_lambda --results_dir=$results_dir \
        --checkpoints_dir=$checkpoints_dir --load_pretrained=$load_pretrained &

        wait
      done
    done
  done
done



echo "||||||||done with training above "$1"|||||||||||||||||||"