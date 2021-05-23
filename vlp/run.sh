CUDA_VISIBLE_DEVICES=1,2 python run_webqa.py --new_segment_ids --do_train --train_batch_size 128 --split train \
--answer_provided_by 'img' --task_to_learn 'filter' --num_workers 8 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 \
--gradient_accumulation_steps 64 --save_loss_curve --num_train_epochs 6 --output_dir tmp/tmp-filter_alone_no_meta --no_img_meta \
--local_rank 0 --global_rank 0 --world_size 2 &

CUDA_VISIBLE_DEVICES=1,2 python run_webqa.py --new_segment_ids --do_train --train_batch_size 128 --split train \
--answer_provided_by 'img' --task_to_learn 'filter' --num_workers 8 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 \
--gradient_accumulation_steps 64 --save_loss_curve --num_train_epochs 6 --output_dir tmp/tmp-filter_alone_no_meta --no_img_meta \
--local_rank 1 --global_rank 1 --world_size 2
#CUDA_VISIBLE_DEVICES=3 python run_webqa.py --new_segment_ids --train_batch_size 16 --split ood_test --answer_provided_by 'img' --task_to_learn 'filter' --num_workers 8 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 8 --save_loss_curve --output_dir tmp/img_filter --recover_ori_ckpt &

#CUDA_VISIBLE_DEVICES=3 python run_webqa.py --new_segment_ids --train_batch_size 16 --split ood_test --answer_provided_by 'img' --task_to_learn 'filter' --num_workers 8 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 8 --save_loss_curve --output_dir tmp/img_filter &

#CUDA_VISIBLE_DEVICES=0 python run_webqa.py --new_segment_ids --train_batch_size 16 --split ind_test --answer_provided_by 'img' --task_to_learn 'filter' --num_workers 8 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 8 --save_loss_curve --output_dir tmp/img_filter --recover_ori_ckpt &

#CUDA_VISIBLE_DEVICES=2 python run_webqa.py --new_segment_ids --train_batch_size 16 --split ind_test --answer_provided_by 'img' --task_to_learn 'filter' --num_workers 8 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 8 --save_loss_curve --output_dir tmp/img_filter &