--local_rank 1 --global_rank 1 --world_size 2
#CUDA_VISIBLE_DEVICES=3 python run_webqa.py --new_segment_ids --train_batch_size 16 --split ood_test --answer_provided_by 'img' --task_to_learn 'filter' --num_workers 8 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 8 --save_loss_curve --output_dir tmp/img_filter --recover_ori_ckpt &

filter infr uni-modal
CUDA_VISIBLE_DEVICES=2 python run_webqa.py --new_segment_ids --train_batch_size 16 --split test --answer_provided_by 'txt' --task_to_learn 'filter' --num_workers 4 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 1 --save_loss_curve --output_dir light_output/filter_txt --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_txt --recover_step 6 &&
CUDA_VISIBLE_DEVICES=3 python run_webqa.py --new_segment_ids --train_batch_size 16 --split test --answer_provided_by 'img' --task_to_learn 'filter' --num_workers 2 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 1 --save_loss_curve --output_dir light_output/filter_img_detectron_upd --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_img_detectron_upd --recover_step 6
CUDA_VISIBLE_DEVICES=3 python run_webqa.py --new_segment_ids --train_batch_size 16 --split test --answer_provided_by 'img' --task_to_learn 'filter' --num_workers 4 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 1 --save_loss_curve --output_dir light_output/filter_both_x_detectron_upd --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_both_x_detectron_upd --recover_step 3 &&
CUDA_VISIBLE_DEVICES=3 python run_webqa.py --new_segment_ids --train_batch_size 16 --split test --answer_provided_by 'txt' --task_to_learn 'filter' --num_workers 4 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 1 --save_loss_curve --output_dir light_output/filter_both_x_detectron_upd --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_both_x_detectron_upd --recover_step 3

CUDA_VISIBLE_DEVICES=3 python run_webqa.py --new_segment_ids --train_batch_size 32 --split test --answer_provided_by 'img' --task_to_learn 'filter' --num_workers 2 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 1 --save_loss_curve --output_dir light_output/filter_img_vinvl_upd --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_img_vinvl_upd --recover_step 4

filter infr both detectron
CUDA_VISIBLE_DEVICES=0 python run_webqa.py --new_segment_ids --train_batch_size 16 --split test --answer_provided_by 'img' --task_to_learn 'filter' --num_workers 4 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 1 --save_loss_curve --output_dir light_output/filter_both_x_detectron_upd --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_both_x_detectron_upd --recover_step 3 --use_x_distractors &&
CUDA_VISIBLE_DEVICES=3 python run_webqa.py --new_segment_ids --train_batch_size 16 --split test --answer_provided_by 'txt' --task_to_learn 'filter' --num_workers 4 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 1 --save_loss_curve --output_dir light_output/filter_both_x_detectron_upd --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_both_x_detectron_upd --recover_step 3 --use_x_distractors &&
CUDA_VISIBLE_DEVICES=3 python run_webqa_vinvl.py --new_segment_ids --train_batch_size 20 --split test --answer_provided_by 'img' --task_to_learn 'filter' --num_workers 4 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 1 --save_loss_curve --output_dir light_output/filter_both_x_vinvl_upd --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_both_x_vinvl_upd --recover_step 1 --use_x_distractors &&
CUDA_VISIBLE_DEVICES=0 python run_webqa_vinvl.py --new_segment_ids --train_batch_size 20 --split test --answer_provided_by 'txt' --task_to_learn 'filter' --num_workers 2 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 1 --save_loss_curve --output_dir light_output/filter_both_x_vinvl_upd --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_both_x_vinvl_upd --recover_step 1 --use_x_distractors &&

filter infr both detectron (partial input)
CUDA_VISIBLE_DEVICES=2 python run_webqa.py --new_segment_ids --train_batch_size 16 --split test --answer_provided_by 'img' --task_to_learn 'filter' --num_workers 4 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 1 --save_loss_curve --output_dir light_output/filter_both_x_detectron --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_both_x_detectron --recover_step 4 --no_img_content &&
CUDA_VISIBLE_DEVICES=1 python run_webqa.py --new_segment_ids --train_batch_size 16 --split test --answer_provided_by 'img' --task_to_learn 'filter' --num_workers 4 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 1 --save_loss_curve --output_dir light_output/filter_both_x_detectron --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_both_x_detectron --recover_step 4 --no_img_meta &&
CUDA_VISIBLE_DEVICES=3 python run_webqa.py --new_segment_ids --train_batch_size 16 --split test --answer_provided_by 'txt' --task_to_learn 'filter' --num_workers 4 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 1 --save_loss_curve --output_dir light_output/filter_both_x_detectron --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_both_x_detectron --recover_step 4 --no_txt_fact


filter infr on val both detectron (for deciding th)
CUDA_VISIBLE_DEVICES=2 python run_webqa.py --new_segment_ids --train_batch_size 16 --split val --answer_provided_by 'img' --task_to_learn 'filter' --num_workers 4 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 1 --save_loss_curve --output_dir light_output/filter_both_x_detectron --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_both_x_detectron --recover_step 4 --use_x_distractors &&
CUDA_VISIBLE_DEVICES=3 python run_webqa.py --new_segment_ids --train_batch_size 16 --split val --answer_provided_by 'txt' --task_to_learn 'filter' --num_workers 4 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 1 --save_loss_curve --output_dir light_output/filter_both_x_detectron --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_both_x_detectron --recover_step 4 --use_x_distractors

filter img detectron
CUDA_VISIBLE_DEVICES=2 python run_webqa.py --new_segment_ids --train_batch_size 128 --split train --do_train --answer_provided_by 'img' --task_to_learn 'filter' --num_workers 4 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 64 --save_loss_curve --output_dir light_output/filter_img_detectron_upd --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_img_detectron_upd --num_train_epochs 9

filter txt
CUDA_VISIBLE_DEVICES=2 python run_webqa.py --new_segment_ids --train_batch_size 128 --split train --answer_provided_by 'txt' --task_to_learn 'filter' --num_workers 4 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 64 --save_loss_curve --do_train --output_dir light_output/filter_txt --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_txt --num_train_epochs 8

filter_both_x_detectron (Herron)
CUDA_VISIBLE_DEVICES=3 python run_webqa.py --new_segment_ids --train_batch_size 128 --split train --answer_provided_by 'img|txt' --task_to_learn 'filter' --num_workers 4 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 128 --save_loss_curve --output_dir light_output/filter_both_x_detectron_upd --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_both_x_detectron_upd --use_x_distractors --do_train --num_train_epochs 6

filter_img_vinvl (Tiger)
CUDA_VISIBLE_DEVICES=3 python run_webqa_vinvl.py --new_segment_ids --train_batch_size 128 --split train --answer_provided_by 'img' --task_to_learn 'filter' --num_workers 4 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 64 --save_loss_curve --output_dir light_output/filter_img_vinvl_upd --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_img_vinvl_upd --do_train --num_train_epochs 9

filter_both_x_vinvl (Herron)
CUDA_VISIBLE_DEVICES=2 python run_webqa_vinvl.py --new_segment_ids --train_batch_size 128 --split train --answer_provided_by 'img|txt' --task_to_learn 'filter' --num_workers 4 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 128 --save_loss_curve --output_dir light_output/filter_both_x_vinvl_upd2 --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_both_x_vinvl_upd2 --use_x_distractors --do_train --num_train_epochs 6

train q-only baseline on full data (Herron)
CUDA_VISIBLE_DEVICES=0 python run_webqa.py --new_segment_ids --do_train --train_batch_size 128 --split train --answer_provided_by 'txt|img' --task_to_learn 'qa' --num_workers 8 --max_pred 50 --mask_prob 0.5 --learning_rate 1e-4 --gradient_accumulation_steps 16 --save_loss_curve --num_train_epochs 10 --output_dir light_output/detectron_both_qa_qonly_upd --ckpts_dir /data/yingshac/MMMHQA/ckpts/detectron_both_qa_qonly_upd --no_img_meta --no_img_content --no_txt_fact &&
CUDA_VISIBLE_DEVICES=1 python run_webqa_vinvl.py --new_segment_ids --do_train --train_batch_size 128 --split train --answer_provided_by 'txt|img' --task_to_learn 'qa' --num_workers 4 --max_pred 50 --mask_prob 0.5 --learning_rate 1e-4 --gradient_accumulation_steps 8 --save_loss_curve --num_train_epochs 16 --output_dir light_output/vinvl_both_qa_qonly --ckpts_dir /data/yingshac/MMMHQA/ckpts/vinvl_both_qa_qonly --no_img_meta --no_img_content --no_txt_fact

train img qa
CUDA_VISIBLE_DEVICES=1 python run_webqa.py --new_segment_ids --do_train --train_batch_size 128 --split train --answer_provided_by 'img' --task_to_learn 'qa' --num_workers 4 --max_pred 50 --mask_prob 0.5 --learning_rate 1e-4 --gradient_accumulation_steps 4 --save_loss_curve --num_train_epochs 16 --output_dir light_output/detectron_img_qa_upd --ckpts_dir /data/yingshac/MMMHQA/ckpts/detectron_img_qa_upd
CUDA_VISIBLE_DEVICES=2 python run_webqa_vinvl.py --new_segment_ids --do_train --train_batch_size 128 --split train --answer_provided_by 'img' --task_to_learn 'qa' --num_workers 4 --max_pred 50 --mask_prob 0.5 --learning_rate 1e-4 --gradient_accumulation_steps 4 --save_loss_curve --num_train_epochs 16 --output_dir light_output/vinvl_img_qa_upd --ckpts_dir /data/yingshac/MMMHQA/ckpts/vinvl_img_qa_upd

#CUDA_VISIBLE_DEVICES=2 python run_webqa.py --new_segment_ids --train_batch_size 16 --split ind_test --answer_provided_by 'img' --task_to_learn 'filter' --num_workers 8 --max_pred 10 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 8 --save_loss_curve --output_dir tmp/img_filter &

#CUDA_VISIBLE_DEVICES=0 python run_webqa.py --new_segment_ids --val_loss --train_batch_size 128 --split val --answer_provided_by 'txt' --task_to_learn 'qa' --num_workers 8 --max_pred 50 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 8 --save_loss_curve --num_train_epochs 5 --output_dir tmp/qa_alone_txt_0528data --recover_step 1 &&


#CUDA_VISIBLE_DEVICES=0 python run_webqa.py --new_segment_ids --val_loss --train_batch_size 128 --split val --answer_provided_by 'img' --task_to_learn 'qa' --num_workers 8 --max_pred 30 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 4 --save_loss_curve --num_train_epochs 5 --output_dir tmp/qa_alone_img_and_cxt --recover_step 1 &&


#CUDA_VISIBLE_DEVICES=0 python run_webqa.py --new_segment_ids --val_loss --train_batch_size 128 --split val --answer_provided_by 'img|txt' --task_to_learn 'qa' --num_workers 8 --max_pred 50 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 4 --save_loss_curve --num_train_epochs 5 --output_dir tmp/qa_full_data --recover_step 1 &&


#CUDA_VISIBLE_DEVICES=0 python run_webqa.py --new_segment_ids --val_loss --train_batch_size 128 --split val --answer_provided_by 'txt' --task_to_learn 'filter' --num_workers 8 --max_pred 50 --mask_prob 1.0 --learning_rate 3e-5 --gradient_accumulation_steps 8 --save_loss_curve --num_train_epochs 5 --output_dir tmp/filter_alone_txt_20choices_0528data --recover_step 10 


