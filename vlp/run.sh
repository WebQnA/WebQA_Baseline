#CUDA_VISIBLE_DEVICES=2,3 python run_webqa.py --new_segment_ids --do_train --train_batch_size 128 --split train \
#--answer_provided_by 'img|txt' --task_to_learn 'filter' --num_workers 8 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 \
#--gradient_accumulation_steps 64 --save_loss_curve --num_train_epochs 5 --output_dir tmp/filter_full_data \
#--local_rank 0 --global_rank 0 --world_size 2 &

#CUDA_VISIBLE_DEVICES=1 python run_webqa.py --new_segment_ids --do_train --train_batch_size 128 --split train --answer_provided_by 'txt' --task_to_learn 'filter' --num_workers 16 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 64 --save_loss_curve --num_train_epochs 6 --output_dir light_output/filter_txt_neg_ranked_by_IoU --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_txt_neg_ranked_by_IoU

CUDA_VISIBLE_DEVICES=3 python run_webqa.py --new_segment_ids --do_train --train_batch_size 128 --split train --answer_provided_by 'txt' --task_to_learn 'filter' --num_workers 8 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 16 --save_loss_curve --num_train_epochs 8 --output_dir light_output/filter_txt_neg_ranked_by_RE_8 --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_txt_neg_ranked_by_RE_8 --txt_dataset_json_path /home/yingshac/CYS/WebQnA/WebQnA_data_new/txt_dataset_0725_ranked_by_RE.json && CUDA_VISIBLE_DEVICES=3 python run_webqa.py --new_segment_ids --do_train --train_batch_size 128 --split train --answer_provided_by 'txt' --task_to_learn 'filter' --num_workers 8 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 32 --save_loss_curve --num_train_epochs 8 --output_dir light_output/filter_txt_neg_ranked_by_RE_16 --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_txt_neg_ranked_by_RE_16 --txt_dataset_json_path /home/yingshac/CYS/WebQnA/WebQnA_data_new/txt_dataset_0725_ranked_by_RE.json --txt_filter_max_choices 16

CUDA_VISIBLE_DEVICES=1 python run_webqa.py --new_segment_ids --do_train --train_batch_size 128 --split train --answer_provided_by 'txt' --task_to_learn 'filter' --num_workers 8 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 16 --save_loss_curve --num_train_epochs 8 --output_dir light_output/filter_txt_neg_ranked_by_IoU_8 --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_txt_neg_ranked_by_IoU_8 && CUDA_VISIBLE_DEVICES=1 python run_webqa.py --new_segment_ids --do_train --train_batch_size 128 --split train --answer_provided_by 'txt' --task_to_learn 'filter' --num_workers 8 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 32 --save_loss_curve --num_train_epochs 8 --output_dir light_output/filter_txt_neg_ranked_by_IoU_16 --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_txt_neg_ranked_by_IoU_16 --txt_filter_max_choices 16


--local_rank 1 --global_rank 1 --world_size 2
#CUDA_VISIBLE_DEVICES=3 python run_webqa.py --new_segment_ids --train_batch_size 16 --split ood_test --answer_provided_by 'img' --task_to_learn 'filter' --num_workers 8 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 8 --save_loss_curve --output_dir tmp/img_filter --recover_ori_ckpt &

filter infr
#CUDA_VISIBLE_DEVICES=2 python run_webqa.py --new_segment_ids --train_batch_size 16 --split ood_test --answer_provided_by 'img' --task_to_learn 'filter' --num_workers 8 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 8 --save_loss_curve --output_dir tmp/img_filter &

filter both with x_distractors
CUDA_VISIBLE_DEVICES=1 python run_webqa_vinvl.py --new_segment_ids --train_batch_size 128 --split ind_test --answer_provided_by 'img|txt' --task_to_learn 'filter' --num_workers 8 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 64 --save_loss_curve --output_dir light_output/filter_both_x_debug --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_both_x_debug --use_x_distractors

CUDA_VISIBLE_DEVICES=0 python run_webqa.py --new_segment_ids --train_batch_size 128 --split train --answer_provided_by 'img|txt' --task_to_learn 'filter' --num_workers 8 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 128 --save_loss_curve --output_dir light_output/filter_both_x_detectron --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_both_x_detectron --use_x_distractors --do_train --num_train_epochs 6

#CUDA_VISIBLE_DEVICES=2 python run_webqa.py --new_segment_ids --train_batch_size 16 --split ind_test --answer_provided_by 'img' --task_to_learn 'filter' --num_workers 8 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 8 --save_loss_curve --output_dir tmp/img_filter &

#CUDA_VISIBLE_DEVICES=0 python run_webqa.py --new_segment_ids --val_loss --train_batch_size 128 --split val --answer_provided_by 'txt' --task_to_learn 'qa' --num_workers 8 --max_pred 50 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 8 --save_loss_curve --num_train_epochs 5 --output_dir tmp/qa_alone_txt_0528data --recover_step 1 &&


#CUDA_VISIBLE_DEVICES=0 python run_webqa.py --new_segment_ids --val_loss --train_batch_size 128 --split val --answer_provided_by 'img' --task_to_learn 'qa' --num_workers 8 --max_pred 30 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 4 --save_loss_curve --num_train_epochs 5 --output_dir tmp/qa_alone_img_and_cxt --recover_step 1 &&


#CUDA_VISIBLE_DEVICES=0 python run_webqa.py --new_segment_ids --val_loss --train_batch_size 128 --split val --answer_provided_by 'img|txt' --task_to_learn 'qa' --num_workers 8 --max_pred 50 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 4 --save_loss_curve --num_train_epochs 5 --output_dir tmp/qa_full_data --recover_step 1 &&


#CUDA_VISIBLE_DEVICES=0 python run_webqa.py --new_segment_ids --val_loss --train_batch_size 128 --split val --answer_provided_by 'txt' --task_to_learn 'filter' --num_workers 8 --max_pred 50 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 8 --save_loss_curve --num_train_epochs 5 --output_dir tmp/filter_alone_txt_20choices_0528data --recover_step 10 


#CUDA_VISIBLE_DEVICES=0 python run_webqa.py --new_segment_ids --val_loss --train_batch_size 128 --split val --answer_provided_by 'img|txt' --task_to_learn 'filter' --num_workers 8 --max_pred 50 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 16 --save_loss_curve --num_train_epochs 5 --output_dir tmp/filter_full_data --recover_step 7 

#CUDA_VISIBLE_DEVICES=3 python run_webqa.py --new_segment_ids --val_loss --train_batch_size 128 --split val --answer_provided_by 'img' --task_to_learn 'filter' --num_workers 8 --max_pred 50 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 8 --save_loss_curve --num_train_epochs 5 --output_dir tmp/tmp-filter_alone_w_img_20chioces_0527 --recover_step 1 &&
#CUDA_VISIBLE_DEVICES=3 python run_webqa.py --new_segment_ids --val_loss --train_batch_size 128 --split val --answer_provided_by 'img' --task_to_learn 'filter' --num_workers 8 --max_pred 50 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 8 --save_loss_curve --num_train_epochs 5 --output_dir tmp/tmp-filter_alone_w_img_20chioces_0527 --recover_step 2 &&
CUDA_VISIBLE_DEVICES=3 python run_webqa.py --new_segment_ids --val_loss --train_batch_size 128 --split val --answer_provided_by 'img' --task_to_learn 'filter' --num_workers 8 --max_pred 50 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 16 --save_loss_curve --num_train_epochs 5 --output_dir tmp/tmp-filter_alone_w_img_20chioces_0527 --recover_step 3 &&
CUDA_VISIBLE_DEVICES=3 python run_webqa.py --new_segment_ids --val_loss --train_batch_size 128 --split val --answer_provided_by 'img' --task_to_learn 'filter' --num_workers 8 --max_pred 50 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 16 --save_loss_curve --num_train_epochs 5 --output_dir tmp/tmp-filter_alone_w_img_20chioces_0527 --recover_step 4 &&
CUDA_VISIBLE_DEVICES=3 python run_webqa.py --new_segment_ids --val_loss --train_batch_size 128 --split val --answer_provided_by 'img' --task_to_learn 'filter' --num_workers 8 --max_pred 50 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 16 --save_loss_curve --num_train_epochs 5 --output_dir tmp/tmp-filter_alone_w_img_20chioces_0527 --recover_step 5