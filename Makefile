train:
	CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore::UserWarning train.py --cfg configs/multi.yaml OUTPUT_DIR /nobackup/le/LoCoNet/loconet RESUME_PATH weights/loconet_ava_best.model

eval:
	python -W ignore::UserWarning test_multicard.py --cfg configs/multi.yaml evalDataType val

eval_landmark_1_1:
	python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 1 layer 1

eval_landmark_1_2:
	python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 1 layer 2

eval_landmark_1_3:
	python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 1 layer 3

eval_landmark_1_4:
	python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 1 layer 4

eval_landmark_2_1:
	python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 2 layer 1

eval_landmark_2_2:
	python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 2 layer 2

eval_landmark_2_3:
	python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 2 layer 3

eval_landmark_2_4:
	python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 2 layer 4

eval_landmark_4_1:
	python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1

eval_landmark_4_2:
	python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 2

eval_landmark_4_3:
	python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 3

eval_landmark_4_4:
	python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 4

eval_landmark_8_1:
	python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 8 layer 1

eval_landmark_8_2:
	python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 8 layer 2

eval_landmark_8_3:
	python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 8 layer 3

eval_landmark_8_4:
	python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 8 layer 4

eval_landmark_16_1:
	python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 16 layer 1

eval_landmark_16_2:
	python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 16 layer 2

eval_landmark_16_3:
	python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 16 layer 3

eval_landmark_16_4:
	python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 16 layer 4

test_reverse:
	python -W ignore::UserWarning test.py --cfg configs/multi.yaml evalDataType test_reverse

test_reverse_landmark_channel_1_layer_1:
	python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 1 layer 1

test_reverse_landmark_channel_1_layer_2:
	python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 1 layer 2

test_reverse_landmark_channel_1_layer_3:
	python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 1 layer 3

test_reverse_landmark_channel_1_layer_4:
	python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 1 layer 4

test_reverse_landmark_channel_2_layer_1:
	python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 2 layer 1

test_reverse_landmark_channel_2_layer_2:
	python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 2 layer 2

test_reverse_landmark_channel_2_layer_3:
	python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 2 layer 3

test_reverse_landmark_channel_2_layer_4:
	python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 2 layer 4

test_reverse_landmark_channel_4_layer_1:
	python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 1

test_reverse_landmark_channel_4_layer_2:
	python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 2

test_reverse_landmark_channel_4_layer_3:
	python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 3

test_reverse_landmark_channel_4_layer_4:
	python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 4

test_reverse_landmark_channel_8_layer_1:
	python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 8 layer 1

test_reverse_landmark_channel_8_layer_2:
	python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 8 layer 2

test_reverse_landmark_channel_8_layer_3:
	python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 8 layer 3

test_reverse_landmark_channel_8_layer_4:
	python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 8 layer 4

test_reverse_landmark_channel_16_layer_1:
	python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 16 layer 1

test_reverse_landmark_channel_16_layer_2:
	python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 16 layer 2

test_reverse_landmark_channel_16_layer_3:
	python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 16 layer 3

test_reverse_landmark_channel_16_layer_4:
	python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 16 layer 4

test_val:
	CUDA_VISIBLE_DEVICES=4 python -W ignore::UserWarning test.py --cfg configs/multi.yaml evalDataType val

test_mute:
	CUDA_VISIBLE_DEVICES=3 python -W ignore::UserWarning test.py --cfg configs/multi.yaml evalDataType test_mute

test_grad_cam:
	CUDA_VISIBLE_DEVICES=5 python -W ignore::UserWarning test_grad_cam.py --cfg configs/multi.yaml evalDataType val

compare_talkNCE_reverse:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p ./talknce_AVA/talknce/0_res.csv

compare_talkNCE_normal:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig.csv -p ./talknce_AVA/talknce/0_res.csv

compare_LoCoNet:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/loconet/0_res.csv

compare_LoCoNet_normal:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig.csv -p /nobackup/le/LoCoNet/loconet/0_res.csv

compare_talkNCE_channel_16_layer_1:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_16_1.csv

compare_talkNCE_channel_16_layer_2:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_16_2.csv

compare_talkNCE_channel_16_layer_3:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_16_3.csv

compare_talkNCE_channel_16_layer_4:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_16_4.csv

compare_talkNCE_channel_8_layer_1:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_8_1.csv

compare_talkNCE_channel_8_layer_2:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_8_2.csv

compare_talkNCE_channel_8_layer_3:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_8_3.csv

compare_talkNCE_channel_8_layer_4:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_8_4.csv

compare_talkNCE_channel_4_layer_1:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_test_reverse_4_1.csv

compare_talkNCE_channel_4_layer_2:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_test_reverse_4_2.csv

compare_talkNCE_channel_4_layer_3:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_3.csv

compare_talkNCE_channel_4_layer_4:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_4.csv

compare_talkNCE_channel_2_layer_1:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_2_1.csv

compare_talkNCE_channel_2_layer_2:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_2_2.csv

compare_talkNCE_channel_2_layer_3:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_2_3.csv

compare_talkNCE_channel_2_layer_4:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_2_4.csv

compare_talkNCE_channel_1_layer_1:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_1_1.csv

compare_talkNCE_channel_1_layer_2:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_1_2.csv

compare_talkNCE_channel_1_layer_3:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_1_3.csv

compare_talkNCE_channel_1_layer_4:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_1_4.csv
	
create_lip_landmarks:
	CUDA_VISIBLE_DEVICES=0 python -W ignore landmark.py --l 0 --r 14 --type train

train_landmark_channel_1_layer_1:
	CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 1 layer 1

train_landmark_channel_1_layer_2:
	CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 1 layer 2

train_landmark_channel_1_layer_3:
	CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 1 layer 3

train_landmark_channel_1_layer_4:
	CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 1 layer 4

train_landmark_channel_2_layer_1:
	CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 2 layer 1

train_landmark_channel_2_layer_0:
	CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 2 layer 0

train_landmark_channel_2_layer_2:
	CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 2 layer 2

train_landmark_channel_2_layer_3:
	CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 2 layer 3

train_landmark_channel_2_layer_4:
	CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 2 layer 4

train_landmark_channel_4_layer_0:
	CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 4 layer 0

train_landmark_channel_4_layer_1:
	CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 4 layer 1

train_landmark_channel_4_layer_2:
	CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 4 layer 2

train_landmark_channel_4_layer_3:
	CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 4 layer 3

train_landmark_channel_4_layer_4:
	CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 4 layer 4

train_landmark_channel_8_layer_1:
	CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 8 layer 1

train_landmark_channel_8_layer_2:
	CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 8 layer 2

train_landmark_channel_8_layer_3:
	CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 8 layer 3 

train_landmark_channel_8_layer_4:
	CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 8 layer 4

train_landmark_channel_16_layer_1:
	CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 16 layer 1

train_landmark_channel_16_layer_2:
	CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 16 layer 2

train_landmark_channel_16_layer_3:
	CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 16 layer 3

train_landmark_channel_16_layer_4:
	CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 16 layer 4

train_landmark_loconet_channel_4_layer_1_kl_0.2:
	CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 4 layer 1 consistency_method kl consistency_lambda 0.2 use_talknce False

train_landmark_loconet_channel_4_layer_1_kl_0.4:
	CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 4 layer 1 consistency_method kl consistency_lambda 0.4 use_talknce False

train_landmark_loconet_channel_4_layer_1_kl_0.6:
	CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 4 layer 1 consistency_method kl consistency_lambda 0.6 use_talknce False

train_landmark_loconet_channel_4_layer_1_kl_0.8:
	CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 4 layer 1 consistency_method kl consistency_lambda 0.8 use_talknce False

train_landmark_loconet_channel_4_layer_1_kl_1:
	CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 4 layer 1 consistency_method kl consistency_lambda 1 use_talknce False

train_landmark_channel_4_layer_1_kl_0.2:
	CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 4 layer 1 consistency_method kl consistency_lambda 0.2

train_landmark_channel_4_layer_1_kl_0.4:
	CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 4 layer 1 consistency_method kl consistency_lambda 0.4

train_landmark_channel_4_layer_1_kl_0.6:
	CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 4 layer 1 consistency_method kl consistency_lambda 0.6

train_landmark_channel_4_layer_1_kl_0.8:
	CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 4 layer 1 consistency_method kl consistency_lambda 0.8

train_landmark_channel_4_layer_1_kl_1:
	CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 4 layer 1 consistency_method kl consistency_lambda 1

train_landmark_channel_4_layer_1_mse_0.2:
	CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 4 layer 1 consistency_method mse consistency_lambda 0.2

train_landmark_channel_4_layer_1_mse_0.4:
	CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 4 layer 1 consistency_method mse consistency_lambda 0.4

train_landmark_channel_4_layer_1_mse_0.6:
	CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 4 layer 1 consistency_method mse consistency_lambda 0.6

train_landmark_channel_4_layer_1_mse_0.8:
	CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 4 layer 1 consistency_method mse consistency_lambda 0.8

train_landmark_channel_4_layer_1_mse_1:
	CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 4 layer 1 consistency_method mse consistency_lambda 1

##################################################################################################################################################

train_landmark_dual_channel_4_layer_1_kl_1:
	CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore::UserWarning train_landmark_loconet_dual.py --cfg configs/multi.yaml n_channel 4 layer 1 consistency_method kl consistency_lambda 1

##################################################################################################################################################

train_landmark_channel_4_layer_2_kl_0.2:
	CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 4 layer 2 consistency_method kl consistency_lambda 0.2

train_landmark_channel_4_layer_2_kl_0.4:
	CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 4 layer 2 consistency_method kl consistency_lambda 0.4

train_landmark_channel_4_layer_2_kl_0.6:
	CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 4 layer 2 consistency_method kl consistency_lambda 0.6

train_landmark_channel_4_layer_2_kl_0.8:
	CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 4 layer 2 consistency_method kl consistency_lambda 0.8

train_landmark_channel_4_layer_2_kl_1:
	CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 4 layer 2 consistency_method kl consistency_lambda 1

train_landmark_channel_4_layer_2_mse_0.2:
	CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 4 layer 2 consistency_method mse consistency_lambda 0.2

train_landmark_channel_4_layer_2_mse_0.4:
	CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 4 layer 2 consistency_method mse consistency_lambda 0.4

train_landmark_channel_4_layer_2_mse_0.6:
	CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 4 layer 2 consistency_method mse consistency_lambda 0.6

train_landmark_channel_4_layer_2_mse_0.8:
	CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 4 layer 2 consistency_method mse consistency_lambda 0.8

train_landmark_channel_4_layer_2_mse_1:
	CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore::UserWarning train_landmark_loconet.py --cfg configs/multi.yaml n_channel 4 layer 2 consistency_method mse consistency_lambda 1

##############################################################################################################################

eval_loconet_4_1_kl_0.2_landmark:
	python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method kl consistency_lambda 0.2 use_landmark True use_talknce False

eval_loconet_4_1_kl_0.2_no_landmark:
	python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method kl consistency_lambda 0.2 use_landmark False use_talknce False

eval_loconet_4_1_kl_0.4_landmark:
	CUDA_VISIBLE_DEVICES=0 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method kl consistency_lambda 0.4 use_landmark True use_talknce False

eval_loconet_4_1_kl_0.4_no_landmark:
	CUDA_VISIBLE_DEVICES=1 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method kl consistency_lambda 0.4 use_landmark False use_talknce False

eval_loconet_4_1_kl_0.6_landmark:
	CUDA_VISIBLE_DEVICES=2 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method kl consistency_lambda 0.6 use_landmark True use_talknce False

eval_loconet_4_1_kl_0.6_no_landmark:
	CUDA_VISIBLE_DEVICES=3 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method kl consistency_lambda 0.6 use_landmark False use_talknce False

eval_loconet_4_1_kl_0.8_landmark:
	python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method kl consistency_lambda 0.8 use_landmark True use_talknce False

eval_loconet_4_1_kl_0.8_no_landmark:
	python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method kl consistency_lambda 0.8 use_landmark False use_talknce False

eval_loconet_4_1_kl_1_landmark:
	CUDA_VISIBLE_DEVICES=4 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method kl consistency_lambda 1 use_landmark True use_talknce False

eval_loconet_4_1_kl_1_no_landmark:
	CUDA_VISIBLE_DEVICES=5 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method kl consistency_lambda 1 use_landmark False use_talknce False

eval_landmark_4_1_kl_0.2_landmark:
	python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method kl consistency_lambda 0.2 use_landmark True

eval_landmark_4_1_kl_0.2_no_landmark:
	python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method kl consistency_lambda 0.2 use_landmark False

eval_landmark_4_1_kl_0.4_landmark:
	CUDA_VISIBLE_DEVICES=0 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method kl consistency_lambda 0.4 use_landmark True

eval_landmark_4_1_kl_0.4_no_landmark:
	CUDA_VISIBLE_DEVICES=1 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method kl consistency_lambda 0.4 use_landmark False

eval_landmark_4_1_kl_0.6_landmark:
	CUDA_VISIBLE_DEVICES=2 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method kl consistency_lambda 0.6 use_landmark True

eval_landmark_4_1_kl_0.6_no_landmark:
	CUDA_VISIBLE_DEVICES=3 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method kl consistency_lambda 0.6 use_landmark False

eval_landmark_4_1_kl_0.8_landmark:
	python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method kl consistency_lambda 0.8 use_landmark True

eval_landmark_4_1_kl_0.8_no_landmark:
	python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method kl consistency_lambda 0.8 use_landmark False

eval_landmark_4_1_kl_1_landmark:
	CUDA_VISIBLE_DEVICES=4 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method kl consistency_lambda 1 use_landmark True

eval_landmark_4_1_kl_1_no_landmark:
	CUDA_VISIBLE_DEVICES=5 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method kl consistency_lambda 1 use_landmark False

eval_landmark_4_1_mse_0.2_landmark:
	CUDA_VISIBLE_DEVICES=0 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method mse consistency_lambda 0.2 use_landmark True

eval_landmark_4_1_mse_0.2_no_landmark:
	CUDA_VISIBLE_DEVICES=1 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method mse consistency_lambda 0.2 use_landmark False

eval_landmark_4_1_mse_0.4_landmark:
	CUDA_VISIBLE_DEVICES=2 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method mse consistency_lambda 0.4 use_landmark True

eval_landmark_4_1_mse_0.4_no_landmark:
	CUDA_VISIBLE_DEVICES=3 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method mse consistency_lambda 0.4 use_landmark False

eval_landmark_4_1_mse_0.6_landmark:
	CUDA_VISIBLE_DEVICES=0 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method mse consistency_lambda 0.6 use_landmark True

eval_landmark_4_1_mse_0.6_no_landmark:
	CUDA_VISIBLE_DEVICES=1 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method mse consistency_lambda 0.6 use_landmark False

eval_landmark_4_1_mse_0.8_landmark:
	CUDA_VISIBLE_DEVICES=2 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method mse consistency_lambda 0.8 use_landmark True

eval_landmark_4_1_mse_0.8_no_landmark:
	CUDA_VISIBLE_DEVICES=3 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method mse consistency_lambda 0.8 use_landmark False

eval_landmark_4_1_mse_1_landmark:
	CUDA_VISIBLE_DEVICES=0 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method mse consistency_lambda 1 use_landmark True

eval_landmark_4_1_mse_1_no_landmark:
	CUDA_VISIBLE_DEVICES=1 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method mse consistency_lambda 1 use_landmark False

eval_landmark_4_2_kl_0.2_landmark:
	python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 2 consistency_method kl consistency_lambda 0.2 use_landmark True

eval_landmark_4_2_kl_0.2_no_landmark:
	python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 2 consistency_method kl consistency_lambda 0.2 use_landmark False

eval_landmark_4_2_kl_0.4_landmark:
	CUDA_VISIBLE_DEVICES=0 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 2 consistency_method kl consistency_lambda 0.4 use_landmark True

eval_landmark_4_2_kl_0.4_no_landmark:
	CUDA_VISIBLE_DEVICES=1 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 2 consistency_method kl consistency_lambda 0.4 use_landmark False

eval_landmark_4_2_kl_0.6_landmark:
	CUDA_VISIBLE_DEVICES=4 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 2 consistency_method kl consistency_lambda 0.6 use_landmark True

eval_landmark_4_2_kl_0.6_no_landmark:
	CUDA_VISIBLE_DEVICES=5 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 2 consistency_method kl consistency_lambda 0.6 use_landmark False

eval_landmark_4_2_kl_0.8_landmark:
	CUDA_VISIBLE_DEVICES=0 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 2 consistency_method kl consistency_lambda 0.8 use_landmark True

eval_landmark_4_2_kl_0.8_no_landmark:
	CUDA_VISIBLE_DEVICES=1 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 2 consistency_method kl consistency_lambda 0.8 use_landmark False

eval_landmark_4_2_kl_1_landmark:
	CUDA_VISIBLE_DEVICES=4 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 2 consistency_method kl consistency_lambda 1 use_landmark True

eval_landmark_4_2_kl_1_no_landmark:
	CUDA_VISIBLE_DEVICES=5 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 2 consistency_method kl consistency_lambda 1 use_landmark False

eval_landmark_4_2_mse_0.2_landmark:
	CUDA_VISIBLE_DEVICES=0 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 2 consistency_method mse consistency_lambda 0.2 use_landmark True

eval_landmark_4_2_mse_0.2_no_landmark:
	CUDA_VISIBLE_DEVICES=1 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 2 consistency_method mse consistency_lambda 0.2 use_landmark False

eval_landmark_4_2_mse_0.4_landmark:
	CUDA_VISIBLE_DEVICES=4 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 2 consistency_method mse consistency_lambda 0.4 use_landmark True

eval_landmark_4_2_mse_0.4_no_landmark:
	CUDA_VISIBLE_DEVICES=5 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 2 consistency_method mse consistency_lambda 0.4 use_landmark False

eval_landmark_4_2_mse_0.6_landmark:
	CUDA_VISIBLE_DEVICES=0 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 2 consistency_method mse consistency_lambda 0.6 use_landmark True

eval_landmark_4_2_mse_0.6_no_landmark:
	CUDA_VISIBLE_DEVICES=1 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 2 consistency_method mse consistency_lambda 0.6 use_landmark False

eval_landmark_4_2_mse_0.8_landmark:
	CUDA_VISIBLE_DEVICES=2 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 2 consistency_method mse consistency_lambda 0.8 use_landmark True

eval_landmark_4_2_mse_0.8_no_landmark:
	CUDA_VISIBLE_DEVICES=3 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 2 consistency_method mse consistency_lambda 0.8 use_landmark False

eval_landmark_4_2_mse_1_landmark:
	CUDA_VISIBLE_DEVICES=0 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 2 consistency_method mse consistency_lambda 1 use_landmark True

eval_landmark_4_2_mse_1_no_landmark:
	CUDA_VISIBLE_DEVICES=1 python -W ignore::UserWarning test_multicard_landmark.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 2 consistency_method mse consistency_lambda 1 use_landmark False

###################################################################################################

test_reverse_loconet_channel_4_layer_1_kl_0.2_landmark:
	CUDA_VISIBLE_DEVICES=1 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 1 consistency_method kl consistency_lambda 0.2 use_landmark True use_talknce False

test_reverse_loconet_channel_4_layer_1_kl_0.2_no_landmark:
	CUDA_VISIBLE_DEVICES=2 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 1 consistency_method kl consistency_lambda 0.2 use_landmark False use_talknce False
  
###################################################################################################

test_reverse_landmark_channel_4_layer_1_kl_0.2_landmark:
	CUDA_VISIBLE_DEVICES=1 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 1 consistency_method kl consistency_lambda 0.2 use_landmark True

test_reverse_landmark_channel_4_layer_1_kl_0.2_no_landmark:
	python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 1 consistency_method kl consistency_lambda 0.2 use_landmark False

test_reverse_landmark_channel_4_layer_1_kl_0.4_landmark:
	CUDA_VISIBLE_DEVICES=4 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 1 consistency_method kl consistency_lambda 0.4 use_landmark True

test_reverse_landmark_channel_4_layer_1_kl_0.4_no_landmark:
	CUDA_VISIBLE_DEVICES=5 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 1 consistency_method kl consistency_lambda 0.4 use_landmark False

test_reverse_landmark_channel_4_layer_1_kl_0.6_landmark:
	CUDA_VISIBLE_DEVICES=6 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 1 consistency_method kl consistency_lambda 0.6 use_landmark True

test_reverse_landmark_channel_4_layer_1_kl_0.6_no_landmark:
	CUDA_VISIBLE_DEVICES=7 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 1 consistency_method kl consistency_lambda 0.6 use_landmark False

test_reverse_landmark_channel_4_layer_1_kl_0.8_landmark:
	python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 1 consistency_method kl consistency_lambda 0.8 use_landmark True

test_reverse_landmark_channel_4_layer_1_kl_0.8_no_landmark:
	python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 1 consistency_method kl consistency_lambda 0.8 use_landmark False

test_reverse_landmark_channel_4_layer_1_kl_1_landmark:
	CUDA_VISIBLE_DEVICES=6 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 1 consistency_method kl consistency_lambda 1 use_landmark True

test_reverse_landmark_channel_4_layer_1_kl_1_no_landmark:
	CUDA_VISIBLE_DEVICES=7 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 1 consistency_method kl consistency_lambda 1 use_landmark False

test_reverse_landmark_channel_4_layer_1_mse_0.2_landmark:
	CUDA_VISIBLE_DEVICES=2 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 1 consistency_method mse consistency_lambda 0.2 use_landmark True

test_reverse_landmark_channel_4_layer_1_mse_0.2_no_landmark:
	CUDA_VISIBLE_DEVICES=3 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 1 consistency_method mse consistency_lambda 0.2 use_landmark False

test_reverse_landmark_channel_4_layer_1_mse_0.4_landmark:
	CUDA_VISIBLE_DEVICES=6 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 1 consistency_method mse consistency_lambda 0.4 use_landmark True

test_reverse_landmark_channel_4_layer_1_mse_0.4_no_landmark:
	CUDA_VISIBLE_DEVICES=7 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 1 consistency_method mse consistency_lambda 0.4 use_landmark False

test_reverse_landmark_channel_4_layer_1_mse_0.6_landmark:
	CUDA_VISIBLE_DEVICES=4 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 1 consistency_method mse consistency_lambda 0.6 use_landmark True

test_reverse_landmark_channel_4_layer_1_mse_0.6_no_landmark:
	CUDA_VISIBLE_DEVICES=5 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 1 consistency_method mse consistency_lambda 0.6 use_landmark False

test_reverse_landmark_channel_4_layer_1_mse_0.8_landmark:
	CUDA_VISIBLE_DEVICES=6 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 1 consistency_method mse consistency_lambda 0.8 use_landmark True

test_reverse_landmark_channel_4_layer_1_mse_0.8_no_landmark:
	CUDA_VISIBLE_DEVICES=7 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 1 consistency_method mse consistency_lambda 0.8 use_landmark False

test_reverse_landmark_channel_4_layer_1_mse_1_landmark:
	CUDA_VISIBLE_DEVICES=2 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 1 consistency_method mse consistency_lambda 1 use_landmark True

test_reverse_landmark_channel_4_layer_1_mse_1_no_landmark:
	CUDA_VISIBLE_DEVICES=3 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 1 consistency_method mse consistency_lambda 1 use_landmark False

test_reverse_landmark_channel_4_layer_2_kl_0.2_landmark:
	python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 2 consistency_method kl consistency_lambda 0.2 use_landmark True

test_reverse_landmark_channel_4_layer_2_kl_0.2_no_landmark:
	python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 2 consistency_method kl consistency_lambda 0.2 use_landmark False

test_reverse_landmark_channel_4_layer_2_kl_0.4_landmark:
	CUDA_VISIBLE_DEVICES=2 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 2 consistency_method kl consistency_lambda 0.4 use_landmark True

test_reverse_landmark_channel_4_layer_2_kl_0.4_no_landmark:
	CUDA_VISIBLE_DEVICES=3 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 2 consistency_method kl consistency_lambda 0.4 use_landmark False

test_reverse_landmark_channel_4_layer_2_kl_0.6_landmark:
	CUDA_VISIBLE_DEVICES=6 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 2 consistency_method kl consistency_lambda 0.6 use_landmark True

test_reverse_landmark_channel_4_layer_2_kl_0.6_no_landmark:
	CUDA_VISIBLE_DEVICES=7 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 2 consistency_method kl consistency_lambda 0.6 use_landmark False

test_reverse_landmark_channel_4_layer_2_kl_0.8_landmark:
	CUDA_VISIBLE_DEVICES=2 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 2 consistency_method kl consistency_lambda 0.8 use_landmark True

test_reverse_landmark_channel_4_layer_2_kl_0.8_no_landmark:
	CUDA_VISIBLE_DEVICES=3 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 2 consistency_method kl consistency_lambda 0.8 use_landmark False

test_reverse_landmark_channel_4_layer_2_kl_1_landmark:
	CUDA_VISIBLE_DEVICES=6 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 2 consistency_method kl consistency_lambda 1 use_landmark True

test_reverse_landmark_channel_4_layer_2_kl_1_no_landmark:
	CUDA_VISIBLE_DEVICES=7 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 2 consistency_method kl consistency_lambda 1 use_landmark False

test_reverse_landmark_channel_4_layer_2_mse_0.2_landmark:
	CUDA_VISIBLE_DEVICES=2 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 2 consistency_method mse consistency_lambda 0.2 use_landmark True

test_reverse_landmark_channel_4_layer_2_mse_0.2_no_landmark:
	CUDA_VISIBLE_DEVICES=3 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 2 consistency_method mse consistency_lambda 0.2 use_landmark False

test_reverse_landmark_channel_4_layer_2_mse_0.4_landmark:
	CUDA_VISIBLE_DEVICES=6 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 2 consistency_method mse consistency_lambda 0.4 use_landmark True

test_reverse_landmark_channel_4_layer_2_mse_0.4_no_landmark:
	CUDA_VISIBLE_DEVICES=7 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 2 consistency_method mse consistency_lambda 0.4 use_landmark False

test_reverse_landmark_channel_4_layer_2_mse_0.6_landmark:
	CUDA_VISIBLE_DEVICES=4 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 2 consistency_method mse consistency_lambda 0.6 use_landmark True

test_reverse_landmark_channel_4_layer_2_mse_0.6_no_landmark:
	CUDA_VISIBLE_DEVICES=5 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 2 consistency_method mse consistency_lambda 0.6 use_landmark False

test_reverse_landmark_channel_4_layer_2_mse_0.8_landmark:
	CUDA_VISIBLE_DEVICES=6 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 2 consistency_method mse consistency_lambda 0.8 use_landmark True

test_reverse_landmark_channel_4_layer_2_mse_0.8_no_landmark:
	CUDA_VISIBLE_DEVICES=7 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 2 consistency_method mse consistency_lambda 0.8 use_landmark False

test_reverse_landmark_channel_4_layer_2_mse_1_landmark:
	CUDA_VISIBLE_DEVICES=2 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 2 consistency_method mse consistency_lambda 1 use_landmark True

test_reverse_landmark_channel_4_layer_2_mse_1_no_landmark:
	CUDA_VISIBLE_DEVICES=3 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType test_reverse n_channel 4 layer 2 consistency_method mse consistency_lambda 1 use_landmark False

##################################################

compare_reverse_loconet_channel_4_layer_1_kl_0.2_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_1_kl_0.2_True_talknce:False.csv

compare_reverse_loconet_channel_4_layer_1_kl_0.2_no_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_1_kl_0.2_False_talknce:False.csv


# test_val_landmark_channel_4_layer_1_kl_0.2_landmark:
# 	python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method kl consistency_lambda 0.2 use_landmark True

# test_val_landmark_channel_4_layer_1_kl_0.2_no_landmark:
# 	python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method kl consistency_lambda 0.2 use_landmark False

# test_val_landmark_channel_4_layer_1_kl_0.8_landmark:
# 	python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method kl consistency_lambda 0.8 use_landmark True

# test_val_landmark_channel_4_layer_1_kl_0.8_no_landmark:
# 	python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method kl consistency_lambda 0.8 use_landmark False

# test_val_landmark_channel_4_layer_1_kl_1_landmark:
# 	python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method kl consistency_lambda 1 use_landmark True

# test_val_landmark_channel_4_layer_1_kl_1_no_landmark:
# 	 python -W ignore::UserWarning test_landmark_loconet.py --cfg configs/multi.yaml evalDataType val n_channel 4 layer 1 consistency_method kl consistency_lambda 1 use_landmark False

compare_reverse_talkNCE_channel_4_layer_1_kl_0.2_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_1_kl_0.2_True.csv

compare_reverse_talkNCE_channel_4_layer_1_kl_0.2_no_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_1_kl_0.2_False.csv

compare_reverse_talkNCE_channel_4_layer_1_kl_0.4_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_1_kl_0.4_True.csv

compare_reverse_talkNCE_channel_4_layer_1_kl_0.4_no_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_1_kl_0.4_False.csv

compare_reverse_talkNCE_channel_4_layer_1_kl_0.6_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_1_kl_0.6_True.csv

compare_reverse_talkNCE_channel_4_layer_1_kl_0.6_no_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_1_kl_0.6_False.csv

compare_reverse_talkNCE_channel_4_layer_1_kl_0.8_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_1_kl_0.8_True.csv

compare_reverse_talkNCE_channel_4_layer_1_kl_0.8_no_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_1_kl_0.8_False.csv

compare_reverse_talkNCE_channel_4_layer_1_kl_1_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_1_kl_1_True.csv

compare_reverse_talkNCE_channel_4_layer_1_kl_1_no_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_1_kl_1_False.csv

compare_reverse_talkNCE_channel_4_layer_1_mse_0.2_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_1_mse_0.2_True.csv

compare_reverse_talkNCE_channel_4_layer_1_mse_0.2_no_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_1_mse_0.2_False.csv

compare_reverse_talkNCE_channel_4_layer_1_mse_0.4_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_1_mse_0.4_True.csv

compare_reverse_talkNCE_channel_4_layer_1_mse_0.4_no_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_1_mse_0.4_False.csv

compare_reverse_talkNCE_channel_4_layer_1_mse_0.6_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_1_mse_0.6_True.csv

compare_reverse_talkNCE_channel_4_layer_1_mse_0.6_no_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_1_mse_0.6_False.csv

compare_reverse_talkNCE_channel_4_layer_1_mse_0.8_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_1_mse_0.8_True.csv

compare_reverse_talkNCE_channel_4_layer_1_mse_0.8_no_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_1_mse_0.8_False.csv

compare_reverse_talkNCE_channel_4_layer_1_mse_1_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_1_mse_1_True.csv

compare_reverse_talkNCE_channel_4_layer_1_mse_1_no_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_1_mse_1_False.csv

# compare_val_talkNCE_channel_4_layer_1_kl_0.2_landmark:
# 	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig.csv -p /nobackup/le/LoCoNet/landmark/0_res_val_4_1_kl_0.2_True.csv

# compare_val_talkNCE_channel_4_layer_1_kl_0.2_no_landmark:
# 	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig.csv -p /nobackup/le/LoCoNet/landmark/0_res_val_4_1_kl_0.2_False.csv

# compare_val_talkNCE_channel_4_layer_1_kl_0.8_landmark:
# 	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig.csv -p /nobackup/le/LoCoNet/landmark/0_res_val_4_1_kl_0.8_True.csv

# compare_val_talkNCE_channel_4_layer_1_kl_0.8_no_landmark:
# 	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig.csv -p /nobackup/le/LoCoNet/landmark/0_res_val_4_1_kl_0.8_False.csv

# compare_val_talkNCE_channel_4_layer_1_kl_1_landmark:
# 	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig.csv -p /nobackup/le/LoCoNet/landmark/0_res_val_4_1_kl_1_True.csv

# compare_val_talkNCE_channel_4_layer_1_kl_1_no_landmark:
# 	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig.csv -p /nobackup/le/LoCoNet/landmark/0_res_val_4_1_kl_1_False.csv

compare_reverse_talkNCE_channel_4_layer_2_kl_0.2_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_2_kl_0.2_True.csv

compare_reverse_talkNCE_channel_4_layer_2_kl_0.2_no_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_2_kl_0.2_False.csv

compare_reverse_talkNCE_channel_4_layer_2_kl_0.4_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_2_kl_0.4_True.csv

compare_reverse_talkNCE_channel_4_layer_2_kl_0.4_no_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_2_kl_0.4_False.csv

compare_reverse_talkNCE_channel_4_layer_2_kl_0.6_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_2_kl_0.6_True.csv

compare_reverse_talkNCE_channel_4_layer_2_kl_0.6_no_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_2_kl_0.6_False.csv

compare_reverse_talkNCE_channel_4_layer_2_kl_0.8_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_2_kl_0.8_True.csv

compare_reverse_talkNCE_channel_4_layer_2_kl_0.8_no_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_2_kl_0.8_False.csv

compare_reverse_talkNCE_channel_4_layer_2_kl_1_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_2_kl_1_True.csv

compare_reverse_talkNCE_channel_4_layer_2_kl_1_no_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_2_kl_1_False.csv

compare_reverse_talkNCE_channel_4_layer_2_mse_0.2_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_2_mse_0.2_True.csv

compare_reverse_talkNCE_channel_4_layer_2_mse_0.2_no_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_2_mse_0.2_False.csv

compare_reverse_talkNCE_channel_4_layer_2_mse_0.4_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_2_mse_0.4_True.csv

compare_reverse_talkNCE_channel_4_layer_2_mse_0.4_no_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_2_mse_0.4_False.csv

compare_reverse_talkNCE_channel_4_layer_2_mse_0.6_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_2_mse_0.6_True.csv

compare_reverse_talkNCE_channel_4_layer_2_mse_0.6_no_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_2_mse_0.6_False.csv

compare_reverse_talkNCE_channel_4_layer_2_mse_0.8_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_2_mse_0.8_True.csv

compare_reverse_talkNCE_channel_4_layer_2_mse_0.8_no_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_2_mse_0.8_False.csv

compare_reverse_talkNCE_channel_4_layer_2_mse_1_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_2_mse_1_True.csv

compare_reverse_talkNCE_channel_4_layer_2_mse_1_no_landmark:
	python utils/get_ava_active_speaker_performance_no_map.py -g /nobackup/le/LoCoNet/AVA_Dataset/csv/val_orig_modified.csv -p /nobackup/le/LoCoNet/landmark/0_res_reverse_4_2_mse_1_False.csv