import sys
sys.path.append("/home/yingshac/CYS/WebQnA/VLP")
from webqa_loader import Preprocess4webqa, webqaDataset
import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer, WhitespaceTokenizer

max_len_a = 400
max_len_b = 109
max_seq_length = max_len_a + max_len_b + 3
len_vis_input = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(
        "bert-base-cased", do_lower_case=False,
        cache_dir='tmp/.pretrained_model_{}'.format(-1))

processor = Preprocess4webqa(3, 0.15, \
    list(tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids, max_len=max_seq_length, \
    len_vis_input=len_vis_input, max_len_a=max_len_a, max_len_b=max_len_b, \
    new_segment_ids=True, \
    truncate_config={'trunc_seg': 'b', 'always_truncate_tail': False}, \
    local_rank=-1)
    
train_dataset = webqaDataset(dataset_json_path="/home/yingshac/CYS/WebQnA/VLP/vlp/tmp/long_data.json", split="train", \
    batch_size=16, tokenizer=tokenizer, gold_feature_folder="/data/yingshac/MMMHQA/imgFeatures_upd/gold", \
    distractor_feature_folder="/data/yingshac/MMMHQA/imgFeatures_upd/distractors", use_num_samples=-1, \
    processor=processor, device=device)

print(train_dataset.__getitem__(0))