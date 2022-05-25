learning_rate=0.001
batch_size=128
ngpu=1
prefetch=4
epochs=100
script=train.py

load_pretrained='snapshots/pretrained'
checkpoints_dir=''

for test_out in 'dtd' 'places'; do
  for seed in 10 20 30; do
    for pi in 0.1; do
      for score in 'OE'; do
        for dataset in 'cifar100' 'cifar10'; do

          results_dir="results_ssnd_${seed}"

          gpu=3
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
done


echo "||||||||done with training above "$1"|||||||||||||||||||"