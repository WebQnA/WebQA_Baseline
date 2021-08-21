img qa val_loss
CUDA_VISIBLE_DEVICES=2 python run_webqa_vinvl.py --new_segment_ids --val_loss --train_batch_size 128 --split val --answer_provided_by 'img' --task_to_learn 'qa' --num_workers 8 --max_pred 50 --mask_prob 0.5 --learning_rate 1e-4 --gradient_accumulation_steps 2 --save_loss_curve --num_train_epochs 5 --output_dir light_output/vinvl_img_qa_sentence --ckpts_dir /data/yingshac/MMMHQA/ckpts/vinvl_img_qa_sentence --recover_step 8 &&

txt qa val_loss

both qa val_loss
CUDA_VISIBLE_DEVICES=2 python run_webqa_vinvl.py --new_segment_ids --val_loss --train_batch_size 128 --split val --answer_provided_by 'img|txt' --task_to_learn 'qa' --num_workers 8 --max_pred 50 --mask_prob 0.5 --learning_rate 1e-4 --gradient_accumulation_steps 1 --save_loss_curve --num_train_epochs 5 --output_dir light_output/vinvl_both_qa_sentence --ckpts_dir /data/yingshac/MMMHQA/ckpts/vinvl_both_qa_sentence --recover_step 8 &&


img filter val_loss
CUDA_VISIBLE_DEVICES=1 python run_webqa.py --new_segment_ids --val_loss --train_batch_size 128 --split val --answer_provided_by 'img' --task_to_learn 'filter' --num_workers 8 --learning_rate 3e-5 --gradient_accumulation_steps 2 --save_loss_curve --num_train_epochs 5 --output_dir light_output/filter_img_detectron --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_img_detectron --recover_step 2 &&

txt filter val_loss
CUDA_VISIBLE_DEVICES=1 python run_webqa.py --new_segment_ids --val_loss --train_batch_size 128 --split val --answer_provided_by 'txt' --task_to_learn 'filter' --num_workers 8 --learning_rate 3e-5 --gradient_accumulation_steps 2 --save_loss_curve --num_train_epochs 5 --output_dir light_output/filter_txt --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_txt --recover_step 2 &&


img qa decoding
CUDA_VISIBLE_DEVICES=1 python decode_webqa_vinvl.py --new_segment_ids --batch_size 32 --answer_provided_by "img" --beam_size 5 --split "ind_test" --num_workers 8 --output_dir light_output/vinvl_img_qa_sentence --ckpts_dir /data/yingshac/MMMHQA/ckpts/vinvl_img_qa_sentence --recover_step 10