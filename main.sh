#!/bin/bash
data_root_dir=/public/data
echo "step 1: create the h5 dataset for coarse segmentation stage...................................................................."
python -u ./datasets/coarse_create_h5.py --data_root_dir=${data_root_dir}
echo "step 2: training the DeepLabv3+ model for coarse segmentation stage............................................................"
for fold_ind in 1 2 3 4 5;do
	python -u train_coarse.py --model=DeepLabv3_plus_skipconnection_3d --fold_ind=${fold_ind} --data_dir=${data_root_dir}/coarse --no-pre_trained --epochs=100 --device=cuda:0 --learning_rate=0.001 --loss=CrossEntropyLoss
done
echo "step 3: training the 3D GCSN model used the pretrained DeepLabv3+ model for coarse segmentation stage.........................."
for fold_ind in 1 2 3 4 5;do
	python -u train_coarse.py --model=DeepLabv3_plus_gcn_skipconnection_3d --fold_ind=${fold_ind} --data_dir=${data_root_dir}/coarse --pre_trained --epochs=50 --device=cuda:0 --learning_rate=0.001 --gcn_mode=2 --ds_weight=0.3 --loss=CrossEntropyLoss
done
echo "step 4: extracting coarse probability maps....................................................................................."
for fold_ind in 1 2 3 4 5;do
	python -u coarse_semantic_feature.py --device=cuda:0 --fold_ind=${fold_ind} --model=DeepLabv3_plus_gcn_skipconnection_3d --data_dir=${data_root_dir}/coarse --gcn_mode=2 --ds_weight=0.3 --loss=CrossEntropyLoss --pre_trained
done
echo "step 5: create the h5 dataset for segmentation refinement stage................................................................"
python -u ./datasets/fine_create_h5.py --coarse_dir=${data_root_dir}/coarse --fine_dir=${data_root_dir}/fine
echo "step 6: training the 2D ResUNet model for refinement stage....................................................................."
for fold_ind in 1 2 3 4 5;do
	python -u train_fine.py --fold_ind=${fold_ind} --data_dir=${data_root_dir}/fine --device=cuda:0
done
echo "step 7: testing................................................................................................................"
for fold_ind in 1 2 3 4 5;do
	python -u test_coarse_fine.py --device=cuda:0 --fold_ind=${fold_ind} --data_dir=${data_root_dir}
done
