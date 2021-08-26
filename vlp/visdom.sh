img qa val_loss
CUDA_VISIBLE_DEVICES=2 python run_webqa_vinvl.py --new_segment_ids --val_loss --train_batch_size 128 --split val --answer_provided_by 'img' --task_to_learn 'qa' --num_workers 8 --max_pred 50 --mask_prob 0.5 --learning_rate 1e-4 --gradient_accumulation_steps 2 --save_loss_curve --num_train_epochs 5 --output_dir light_output/vinvl_img_qa_sentence --ckpts_dir /data/yingshac/MMMHQA/ckpts/vinvl_img_qa_sentence --recover_step 8 &&

txt qa val_loss

both qa val_loss
CUDA_VISIBLE_DEVICES=2 python run_webqa_vinvl.py --new_segment_ids --val_loss --train_batch_size 128 --split val --answer_provided_by 'img|txt' --task_to_learn 'qa' --num_workers 8 --max_pred 50 --mask_prob 0.5 --learning_rate 1e-4 --gradient_accumulation_steps 1 --save_loss_curve --num_train_epochs 5 --output_dir light_output/vinvl_both_qa_sentence --ckpts_dir /data/yingshac/MMMHQA/ckpts/vinvl_both_qa_sentence --recover_step 8 &&


img filter val_loss
CUDA_VISIBLE_DEVICES=1 python run_webqa.py --new_segment_ids --val_loss --train_batch_size 128 --split val --answer_provided_by 'img' --task_to_learn 'filter' --num_workers 8 --learning_rate 3e-5 --gradient_accumulation_steps 2 --save_loss_curve --num_train_epochs 5 --output_dir light_output/filter_img_detectron --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_img_detectron --recover_step 2 &&

txt filter val_loss
CUDA_VISIBLE_DEVICES=1 python run_webqa.py --new_segment_ids --val_loss --train_batch_size 128 --split val --answer_provided_by 'txt' --task_to_learn 'filter' --num_workers 8 --learning_rate 3e-5 --gradient_accumulation_steps 2 --save_loss_curve --num_train_epochs 5 --output_dir light_output/filter_txt --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_txt --recover_step 2 &&

both filter val_loss
CUDA_VISIBLE_DEVICES=2 python run_webqa.py --new_segment_ids --val_loss --train_batch_size 128 --split val --answer_provided_by 'img|txt' --task_to_learn 'filter' --num_workers 4 --learning_rate 3e-5 --gradient_accumulation_steps 4 --save_loss_curve --num_train_epochs 5 --output_dir light_output/filter_both_x_detectron --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_both_x_detectron --recover_step 7 --use_x_distractors &&
CUDA_VISIBLE_DEVICES=2 python run_webqa.py --new_segment_ids --val_loss --train_batch_size 128 --split val --answer_provided_by 'img|txt' --task_to_learn 'filter' --num_workers 4 --learning_rate 3e-5 --gradient_accumulation_steps 4 --save_loss_curve --num_train_epochs 5 --output_dir light_output/filter_both_x_vinvl --ckpts_dir /data/yingshac/MMMHQA/ckpts/filter_both_x_vinvl --recover_step 8 --use_x_distractors &&


img qa decoding
CUDA_VISIBLE_DEVICES=1 python decode_webqa_vinvl.py --new_segment_ids --batch_size 32 --answer_provided_by "img" --beam_size 5 --split "test" --num_workers 4 --output_dir light_output/vinvl_img_qa_sentence --ckpts_dir /data/yingshac/MMMHQA/ckpts/vinvl_img_qa_sentence --no_eval --recover_step 10 &&
CUDA_VISIBLE_DEVICES=2 python decode_webqa.py --new_segment_ids --batch_size 32 --answer_provided_by "img" --beam_size 5 --split "test" --num_workers 4 --output_dir light_output/detectron_img_qa_sentence --ckpts_dir /data/yingshac/MMMHQA/ckpts/detectron_img_qa_sentence --no_eval --recover_step 10 &&
CUDA_VISIBLE_DEVICES=0 python decode_webqa.py --new_segment_ids --batch_size 32 --answer_provided_by "img" --beam_size 5 --split "test" --num_workers 4 --output_dir light_output/detectron_both_qa_sentence --ckpts_dir /data/yingshac/MMMHQA/ckpts/detectron_both_qa_sentence --no_eval --recover_step 12 &&
CUDA_VISIBLE_DEVICES=0 python decode_webqa_vinvl.py --new_segment_ids --batch_size 32 --answer_provided_by "img" --beam_size 5 --split "test" --num_workers 4 --output_dir light_output/vinvl_both_qa_sentence --ckpts_dir /data/yingshac/MMMHQA/ckpts/vinvl_both_qa_sentence --no_eval --recover_step 10 &&

img qa decoding e2e
CUDA_VISIBLE_DEVICES=1 python decode_webqa.py --new_segment_ids --batch_size 32 --answer_provided_by "img" --beam_size 5 --split "test" --num_workers 4 --output_dir light_output/detectron_both_qa_sentence --ckpts_dir /data/yingshac/MMMHQA/ckpts/detectron_both_qa_sentence --no_eval --recover_step 12 --img_dataset_json_path /home/yingshac/CYS/WebQnA/VLP/vlp/light_output/filter_both_x_detectron/pred_dataset_th25_test_-1_step4_img_16_True_True_img_dataset_0823_clean_te_UNknown_modality.json
CUDA_VISIBLE_DEVICES=1 python decode_webqa_vinvl.py --new_segment_ids --batch_size 32 --answer_provided_by "img" --beam_size 5 --split "test" --num_workers 4 --output_dir light_output/vinvl_both_qa_sentence --ckpts_dir /data/yingshac/MMMHQA/ckpts/vinvl_both_qa_sentence --no_eval --recover_step 10 --img_dataset_json_path /home/yingshac/CYS/WebQnA/VLP/vlp/light_output/filter_both_x_vinvl/pred_dataset_th25_test_-1_step3_img_16_True_True_img_dataset_0823_clean_te_UNknown_modality.json

img qa decoding partial input
CUDA_VISIBLE_DEVICES=0 python decode_webqa.py --new_segment_ids --batch_size 32 --answer_provided_by "img" --beam_size 5 --split "test" --num_workers 4 --output_dir light_output/detectron_both_qa_sentence --ckpts_dir /data/yingshac/MMMHQA/ckpts/detectron_both_qa_sentence --no_eval --recover_step 12 --no_img_meta &&
CUDA_VISIBLE_DEVICES=2 python decode_webqa.py --new_segment_ids --batch_size 32 --answer_provided_by "img" --beam_size 5 --split "test" --num_workers 4 --output_dir light_output/detectron_both_qa_sentence --ckpts_dir /data/yingshac/MMMHQA/ckpts/detectron_both_qa_sentence --no_eval --recover_step 12 --no_img_content &&
CUDA_VISIBLE_DEVICES=2 python decode_webqa.py --new_segment_ids --batch_size 32 --answer_provided_by "img" --beam_size 5 --split "test" --num_workers 4 --output_dir light_output/detectron_both_qa_sentence --ckpts_dir /data/yingshac/MMMHQA/ckpts/detectron_both_qa_sentence --no_eval --recover_step 12 --no_img_meta --no_img_content &&

img qa decoding qonly
CUDA_VISIBLE_DEVICES=1 python decode_webqa_vinvl.py --new_segment_ids --batch_size 32 --answer_provided_by "img" --beam_size 5 --split "test" --num_workers 4 --output_dir light_output/vinvl_both_qa_qonly --ckpts_dir /data/yingshac/MMMHQA/ckpts/vinvl_both_qa_qonly --recover_step 6 --no_eval --no_img_meta --no_img_content --no_txt_fact
CUDA_VISIBLE_DEVICES=1 python decode_webqa.py --new_segment_ids --batch_size 32 --answer_provided_by "img" --beam_size 5 --split "test" --num_workers 4 --output_dir light_output/detectron_both_qa_qonly --ckpts_dir /data/yingshac/MMMHQA/ckpts/detectron_both_qa_qonly --recover_step 6 --no_eval --no_img_meta --no_img_content --no_txt_fact


txt qa decoding
CUDA_VISIBLE_DEVICES=3 python decode_webqa.py --new_segment_ids --batch_size 32 --answer_provided_by "txt" --beam_size 5 --split "test" --num_workers 4 --output_dir light_output/txt_qa --ckpts_dir /data/yingshac/MMMHQA/ckpts/txt_qa --no_eval --recover_step 10 &&
CUDA_VISIBLE_DEVICES=2 python decode_webqa.py --new_segment_ids --batch_size 32 --answer_provided_by "txt" --beam_size 5 --split "test" --num_workers 4 --output_dir light_output/detectron_both_qa_sentence --ckpts_dir /data/yingshac/MMMHQA/ckpts/detectron_both_qa_sentence --no_eval --recover_step 12 &&
CUDA_VISIBLE_DEVICES=2 python decode_webqa_vinvl.py --new_segment_ids --batch_size 32 --answer_provided_by "txt" --beam_size 5 --split "test" --num_workers 4 --output_dir light_output/vinvl_both_qa_sentence --ckpts_dir /data/yingshac/MMMHQA/ckpts/vinvl_both_qa_sentence --no_eval --recover_step 10 &&

txt qa decoding e2e
CUDA_VISIBLE_DEVICES=1 python decode_webqa.py --new_segment_ids --batch_size 32 --answer_provided_by "txt" --beam_size 5 --split "test" --num_workers 4 --output_dir light_output/detectron_both_qa_sentence --ckpts_dir /data/yingshac/MMMHQA/ckpts/detectron_both_qa_sentence --no_eval --recover_step 12 --txt_dataset_json_path /home/yingshac/CYS/WebQnA/VLP/vlp/light_output/filter_both_x_detectron/pred_dataset_th25_test_-1_step4_txt_16_True_txt_dataset_0823_clean_te_UNknown_modality.json
CUDA_VISIBLE_DEVICES=0 python decode_webqa_vinvl.py --new_segment_ids --batch_size 32 --answer_provided_by "txt" --beam_size 5 --split "test" --num_workers 4 --output_dir light_output/vinvl_both_qa_sentence --ckpts_dir /data/yingshac/MMMHQA/ckpts/vinvl_both_qa_sentence --no_eval --recover_step 10 --txt_dataset_json_path /home/yingshac/CYS/WebQnA/VLP/vlp/light_output/filter_both_x_vinvl/pred_dataset_th25_test_-1_step3_txt_16_True_txt_dataset_0823_clean_te_UNknown_modality.json

txt qa decoding partial input
CUDA_VISIBLE_DEVICES=3 python decode_webqa.py --new_segment_ids --batch_size 32 --answer_provided_by "txt" --beam_size 5 --split "test" --num_workers 4 --output_dir light_output/detectron_both_qa_sentence --ckpts_dir /data/yingshac/MMMHQA/ckpts/detectron_both_qa_sentence --no_eval --recover_step 12 --no_txt_fact &&

txt qa decoding qonly
CUDA_VISIBLE_DEVICES=2 python decode_webqa_vinvl.py --new_segment_ids --batch_size 32 --answer_provided_by "txt" --beam_size 5 --split "test" --num_workers 4 --output_dir light_output/vinvl_both_qa_qonly --ckpts_dir /data/yingshac/MMMHQA/ckpts/vinvl_both_qa_qonly --recover_step 6 --no_eval --no_img_meta --no_img_content --no_txt_fact &&
CUDA_VISIBLE_DEVICES=2 python decode_webqa.py --new_segment_ids --batch_size 32 --answer_provided_by "txt" --beam_size 5 --split "test" --num_workers 4 --output_dir light_output/detectron_both_qa_qonly --ckpts_dir /data/yingshac/MMMHQA/ckpts/detectron_both_qa_qonly --recover_step 6 --no_eval --no_img_meta --no_img_content --no_txt_fact


qa eval
CUDA_VISIBLE_DEVICES=1 python eval.py --mod "img" --file detectron_both_qa_sentence/qa_infr/test_qainfr_no_eval_-1_beam5_img_True_True_step12_img_dataset_0823_clean_te.tsv &&
CUDA_VISIBLE_DEVICES=1 python eval.py --mod "txt" --file detectron_both_qa_sentence/qa_infr/test_qainfr_no_eval_-1_beam5_txt_True_step12_txt_dataset_0823_clean_te.tsv &&
CUDA_VISIBLE_DEVICES=1 python eval.py --mod "txt" --file detectron_both_qa_qonly/qa_infr/test_qainfr_no_eval_-1_beam5_txt_False_step6_txt_dataset_0823_clean_te.tsv
CUDA_VISIBLE_DEVICES=1 python eval.py --mod "img" --file detectron_both_qa_qonly/qa_infr/test_qainfr_no_eval_-1_beam5_img_False_False_step6_img_dataset_0823_clean_te.tsv &&
CUDA_VISIBLE_DEVICES=1 python eval.py --mod "txt" --file txt_qa/qa_infr/test_qainfr_no_eval_-1_beam5_txt_True_step10_txt_dataset_0823_clean_te.tsv &&
CUDA_VISIBLE_DEVICES=1 python eval.py --mod "img" --file detectron_img_qa_sentence/qa_infr/test_qainfr_no_eval_-1_beam5_img_True_True_step10_img_dataset_0823_clean_te.tsv &&
CUDA_VISIBLE_DEVICES=1 python eval.py --mod "img" --file vinvl_img_qa_sentence/qa_infr/test_qainfr_no_eval_-1_beam5_img_True_True_step10_img_dataset_0823_clean_te.tsv &&
CUDA_VISIBLE_DEVICES=1 python eval.py --mod "txt" --file vinvl_both_qa_sentence/qa_infr/test_qainfr_no_eval_-1_beam5_txt_True_step10_txt_dataset_0823_clean_te.tsv &&
CUDA_VISIBLE_DEVICES=1 python eval.py --mod "img" --file vinvl_both_qa_sentence/qa_infr/test_qainfr_no_eval_-1_beam5_img_True_True_step10_img_dataset_0823_clean_te.tsv &&
CUDA_VISIBLE_DEVICES=1 python eval.py --mod "txt" --file vinvl_both_qa_qonly/qa_infr/test_qainfr_no_eval_-1_beam5_txt_False_step6_txt_dataset_0823_clean_te.tsv
CUDA_VISIBLE_DEVICES=1 python eval.py --mod "img" --file vinvl_both_qa_qonly/qa_infr/test_qainfr_no_eval_-1_beam5_img_False_False_step6_img_dataset_0823_clean_te.tsv &&

qa_eval Qcate_breakdown 
CUDA_VISIBLE_DEVICES=1 python eval.py --mod "img" --file detectron_both_qa_sentence/qa_infr/test_qainfr_no_eval_-1_beam5_img_True_True_step12_img_dataset_0823_clean_te.tsv --Qcate_breakdown '["color"]' &&
CUDA_VISIBLE_DEVICES=1 python eval.py --mod "img" --file detectron_both_qa_sentence/qa_infr/test_qainfr_no_eval_-1_beam5_img_True_True_step12_img_dataset_0823_clean_te.tsv --Qcate_breakdown '["shape"]' &&
CUDA_VISIBLE_DEVICES=1 python eval.py --mod "img" --file detectron_both_qa_sentence/qa_infr/test_qainfr_no_eval_-1_beam5_img_True_True_step12_img_dataset_0823_clean_te.tsv --Qcate_breakdown '["number"]' &&
CUDA_VISIBLE_DEVICES=1 python eval.py --mod "img" --file detectron_both_qa_sentence/qa_infr/test_qainfr_no_eval_-1_beam5_img_True_True_step12_img_dataset_0823_clean_te.tsv --Qcate_breakdown '["YesNo"]' &&
CUDA_VISIBLE_DEVICES=1 python eval.py --mod "img" --file detectron_both_qa_sentence/qa_infr/test_qainfr_no_eval_-1_beam5_img_True_True_step12_img_dataset_0823_clean_te.tsv --Qcate_breakdown '["choose"]' &&
CUDA_VISIBLE_DEVICES=1 python eval.py --mod "img" --file detectron_both_qa_sentence/qa_infr/test_qainfr_no_eval_-1_beam5_img_True_True_step12_img_dataset_0823_clean_te.tsv --Qcate_breakdown '["Others"]' &&

CUDA_VISIBLE_DEVICES=1 python eval.py --mod "img" --file detectron_both_qa_qonly/qa_infr/test_qainfr_no_eval_-1_beam5_img_False_False_step6_img_dataset_0823_clean_te.tsv --Qcate_breakdown '["color"]' &&
CUDA_VISIBLE_DEVICES=1 python eval.py --mod "img" --file detectron_both_qa_qonly/qa_infr/test_qainfr_no_eval_-1_beam5_img_False_False_step6_img_dataset_0823_clean_te.tsv --Qcate_breakdown '["shape"]' &&
CUDA_VISIBLE_DEVICES=1 python eval.py --mod "img" --file detectron_both_qa_qonly/qa_infr/test_qainfr_no_eval_-1_beam5_img_False_False_step6_img_dataset_0823_clean_te.tsv --Qcate_breakdown '["number"]' &&
CUDA_VISIBLE_DEVICES=1 python eval.py --mod "img" --file detectron_both_qa_qonly/qa_infr/test_qainfr_no_eval_-1_beam5_img_False_False_step6_img_dataset_0823_clean_te.tsv --Qcate_breakdown '["YesNo"]' &&
CUDA_VISIBLE_DEVICES=1 python eval.py --mod "img" --file detectron_both_qa_qonly/qa_infr/test_qainfr_no_eval_-1_beam5_img_False_False_step6_img_dataset_0823_clean_te.tsv --Qcate_breakdown '["choose"]' &&
CUDA_VISIBLE_DEVICES=1 python eval.py --mod "img" --file detectron_both_qa_qonly/qa_infr/test_qainfr_no_eval_-1_beam5_img_False_False_step6_img_dataset_0823_clean_te.tsv --Qcate_breakdown '["Others"]' &&

CUDA_VISIBLE_DEVICES=1 python eval.py --mod "img" --file detectron_img_qa_sentence/qa_infr/test_qainfr_no_eval_-1_beam5_img_True_True_step10_img_dataset_0823_clean_te.tsv --Qcate_breakdown '["color"]' &&
CUDA_VISIBLE_DEVICES=1 python eval.py --mod "img" --file detectron_img_qa_sentence/qa_infr/test_qainfr_no_eval_-1_beam5_img_True_True_step10_img_dataset_0823_clean_te.tsv --Qcate_breakdown '["shape"]' && 
CUDA_VISIBLE_DEVICES=1 python eval.py --mod "img" --file detectron_img_qa_sentence/qa_infr/test_qainfr_no_eval_-1_beam5_img_True_True_step10_img_dataset_0823_clean_te.tsv --Qcate_breakdown '["number"]' && 
CUDA_VISIBLE_DEVICES=1 python eval.py --mod "img" --file detectron_img_qa_sentence/qa_infr/test_qainfr_no_eval_-1_beam5_img_True_True_step10_img_dataset_0823_clean_te.tsv --Qcate_breakdown '["YesNo"]' && 
CUDA_VISIBLE_DEVICES=1 python eval.py --mod "img" --file detectron_img_qa_sentence/qa_infr/test_qainfr_no_eval_-1_beam5_img_True_True_step10_img_dataset_0823_clean_te.tsv --Qcate_breakdown '["choose"]' && 
CUDA_VISIBLE_DEVICES=1 python eval.py --mod "img" --file detectron_img_qa_sentence/qa_infr/test_qainfr_no_eval_-1_beam5_img_True_True_step10_img_dataset_0823_clean_te.tsv --Qcate_breakdown '["Others"]' && 

qa eval partial input
CUDA_VISIBLE_DEVICES=1 python eval.py --mod "img" --file detectron_both_qa_sentence/qa_infr/test_qainfr_no_eval_-1_beam5_img_True_False_step12_img_dataset_0823_clean_te.tsv &&
CUDA_VISIBLE_DEVICES=1 python eval.py --mod "img" --file detectron_both_qa_sentence/qa_infr/test_qainfr_no_eval_-1_beam5_img_False_True_step12_img_dataset_0823_clean_te.tsv &&
CUDA_VISIBLE_DEVICES=1 python eval.py --mod "img" --file detectron_both_qa_sentence/qa_infr/test_qainfr_no_eval_-1_beam5_img_False_False_step12_img_dataset_0823_clean_te.tsv &&
CUDA_VISIBLE_DEVICES=1 python eval.py --mod "txt" --file detectron_both_qa_sentence/qa_infr/test_qainfr_no_eval_-1_beam5_txt_False_step12_txt_dataset_0823_clean_te.tsv &&