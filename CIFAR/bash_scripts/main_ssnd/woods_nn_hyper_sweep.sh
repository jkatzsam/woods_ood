learning_rate=0.001
batch_size=128
ngpu=1
prefetch=4
epochs=100
script=train.py

load_pretrained='snapshots/pretrained'
checkpoints_dir=''

in_constraint_weight=1
ce_constraint_weight=1
out_constraint_weight=1
constraint_tol=0.05

for test_out in 'dtd' 'places'; do
  for seed in 10 20 30; do
    for pi in 0.1; do
      for score in 'woods_nn'; do
        for dataset in 'cifar100' 'cifar10'; do

          results_dir="results_ssnd_${seed}"

          gpu=0
          lr_lam=0.1
          penalty_mult=1.5
          echo "running $score with $dataset, $test_out, out_constraint_weight $out_constraint_weight, penalty_mult $penalty_mult lr_lam $lr_lam pi=$pi on GPU $gpu"
          CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu CUDA_LAUNCH_BLOCKING=1 python $script $dataset \
          --model wrn --score $score --seed $seed --learning_rate $learning_rate --epochs $epochs \
          --test_out_dataset $test_out --aux_out_dataset $test_out --batch_size=$batch_size --ngpu=$ngpu \
          --prefetch=$prefetch --pi=$pi  --results_dir=$results_dir --lr_lam=$lr_lam \
          --penalty_mult=$penalty_mult --out_constraint_weight=$out_constraint_weight \
          --constraint_tol=$constraint_tol \
          --in_constraint_weight=$in_constraint_weight --ce_constraint_weight=$ce_constraint_weight \
          --checkpoints_dir=$checkpoints_dir --load_pretrained=$load_pretrained &

          gpu=1
          lr_lam=1
          penalty_mult=1.5
          echo "running $score with $dataset, $test_out, out_constraint_weight $out_constraint_weight, penalty_mult $penalty_mult lr_lam $lr_lam pi=$pi on GPU $gpu"
          CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu CUDA_LAUNCH_BLOCKING=1 python $script $dataset \
          --model wrn --score $score --seed $seed --learning_rate $learning_rate --epochs $epochs \
          --test_out_dataset $test_out --aux_out_dataset $test_out --batch_size=$batch_size --ngpu=$ngpu \
          --prefetch=$prefetch --pi=$pi  --results_dir=$results_dir --lr_lam=$lr_lam \
          --penalty_mult=$penalty_mult --out_constraint_weight=$out_constraint_weight \
          --constraint_tol=$constraint_tol \
          --in_constraint_weight=$in_constraint_weight --ce_constraint_weight=$ce_constraint_weight \
          --checkpoints_dir=$checkpoints_dir --load_pretrained=$load_pretrained &

          gpu=2
          lr_lam=2
          penalty_mult=1.5
          echo "running $score with $dataset, $test_out, out_constraint_weight $out_constraint_weight, penalty_mult $penalty_mult lr_lam $lr_lam pi=$pi on GPU $gpu"
          CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu CUDA_LAUNCH_BLOCKING=1 python $script $dataset \
          --model wrn --score $score --seed $seed --learning_rate $learning_rate --epochs $epochs \
          --test_out_dataset $test_out --aux_out_dataset $test_out --batch_size=$batch_size --ngpu=$ngpu \
          --prefetch=$prefetch --pi=$pi  --results_dir=$results_dir --lr_lam=$lr_lam \
          --penalty_mult=$penalty_mult --out_constraint_weight=$out_constraint_weight \
          --constraint_tol=$constraint_tol \
          --in_constraint_weight=$in_constraint_weight --ce_constraint_weight=$ce_constraint_weight \
          --checkpoints_dir=$checkpoints_dir --load_pretrained=$load_pretrained &

          gpu=3
          lr_lam=0.1
          penalty_mult=1.1
          echo "running $score with $dataset, $test_out, out_constraint_weight $out_constraint_weight, penalty_mult $penalty_mult lr_lam $lr_lam pi=$pi on GPU $gpu"
          CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu CUDA_LAUNCH_BLOCKING=1 python $script $dataset \
          --model wrn --score $score --seed $seed --learning_rate $learning_rate --epochs $epochs \
          --test_out_dataset $test_out --aux_out_dataset $test_out --batch_size=$batch_size --ngpu=$ngpu \
          --prefetch=$prefetch --pi=$pi  --results_dir=$results_dir --lr_lam=$lr_lam \
          --penalty_mult=$penalty_mult --out_constraint_weight=$out_constraint_weight \
          --constraint_tol=$constraint_tol \
          --in_constraint_weight=$in_constraint_weight --ce_constraint_weight=$ce_constraint_weight \
          --checkpoints_dir=$checkpoints_dir --load_pretrained=$load_pretrained &

          gpu=4
          lr_lam=1
          penalty_mult=1.1
          echo "running $score with $dataset, $test_out, out_constraint_weight $out_constraint_weight, penalty_mult $penalty_mult lr_lam $lr_lam pi=$pi on GPU $gpu"
          CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu CUDA_LAUNCH_BLOCKING=1 python $script $dataset \
          --model wrn --score $score --seed $seed --learning_rate $learning_rate --epochs $epochs \
          --test_out_dataset $test_out --aux_out_dataset $test_out --batch_size=$batch_size --ngpu=$ngpu \
          --prefetch=$prefetch --pi=$pi  --results_dir=$results_dir --lr_lam=$lr_lam \
          --penalty_mult=$penalty_mult --out_constraint_weight=$out_constraint_weight \
          --constraint_tol=$constraint_tol \
          --in_constraint_weight=$in_constraint_weight --ce_constraint_weight=$ce_constraint_weight \
          --checkpoints_dir=$checkpoints_dir --load_pretrained=$load_pretrained &

          gpu=5
          lr_lam=2
          penalty_mult=1.1
          echo "running $score with $dataset, $test_out, out_constraint_weight $out_constraint_weight, penalty_mult $penalty_mult lr_lam $lr_lam pi=$pi on GPU $gpu"
          CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu CUDA_LAUNCH_BLOCKING=1 python $script $dataset \
          --model wrn --score $score --seed $seed --learning_rate $learning_rate --epochs $epochs \
          --test_out_dataset $test_out --aux_out_dataset $test_out --batch_size=$batch_size --ngpu=$ngpu \
          --prefetch=$prefetch --pi=$pi  --results_dir=$results_dir --lr_lam=$lr_lam \
          --penalty_mult=$penalty_mult --out_constraint_weight=$out_constraint_weight \
          --constraint_tol=$constraint_tol \
          --in_constraint_weight=$in_constraint_weight --ce_constraint_weight=$ce_constraint_weight \
          --checkpoints_dir=$checkpoints_dir --load_pretrained=$load_pretrained &




          wait
        done
      done
    done
  done
done





echo "||||||||done with training above "$1"|||||||||||||||||||"