import gc
import re, os, random, logging, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoModel
from transformers import BertModel, BertTokenizer, DistilBertTokenizer
from pprint import pprint
import sys


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

transformer_model = 'bert-base-cased'
output_name = transformer_model
epochs = 1
ftbert = False

if ftbert: output_name = "ft" + output_name
logger.info('transformer_model={}\nepochs={}'.format(transformer_model, epochs))


class BERT_Arch_Simple(nn.Module):

    def __init__(self, bert):
        super(BERT_Arch_Simple, self).__init__()
        self.bert = BertModel.from_pretrained(transformer_model)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        output = self.bert(sent_id, attention_mask=mask, output_hidden_states=True)
        # all_layers  = [13, batch_size, max_length (36 or 64), 768]
        x = output[1] # [batch_size, 768]
        del output
        gc.collect()
        torch.cuda.empty_cache()
        # logger.info(x.size())
        #x = self.fc(self.dropout(x))
        x = self.fc(x)
        return self.softmax(x)

ckpt_dir = "/data/yingshac/WebQA/CLIP_retrieval_experiments/answer_modality_cls"
data_dir = "/home/yingshac/CYS/WebQnA/WebQnA_data_new/answer_modality_cls"

def read_dataset(path):
    # data csv should contain two columns: 'question' and 'label' (1: img, 0: txt)
    print(path)
    data = pd.read_csv(path)

    #data = data.loc[0:9599,:]
    logger.info("len(data) = {}".format(len(data)))
    return data['question'].tolist(), data['label']


def data_process(data, labels):
    input_ids = []
    attention_masks = []
    bert_tokenizer = BertTokenizer.from_pretrained(transformer_model)
    for sentence in data:
        bert_inp = bert_tokenizer.__call__(json.loads(sentence), max_length=64,
                                           padding='max_length', pad_to_max_length=True,
                                           truncation=True, return_token_type_ids=False)
        input_ids.append(bert_inp['input_ids'])
        attention_masks.append(bert_inp['attention_mask'])

    input_ids = np.asarray(input_ids)
    attention_masks = np.array(attention_masks)
    labels = np.array(labels)
    return input_ids, attention_masks, labels

def load_and_process(path):
    data, labels = read_dataset(path)
    num_of_labels = len(labels.unique())
    logger.info(f'num_of_labels = {num_of_labels}')
    input_ids, attention_masks, labels = data_process(data, labels)

    return input_ids, attention_masks, labels

# function to train the model
def train():
    model.train()

    total_loss, total_accuracy = 0, 0

    # empty list to save model predictions
    total_preds = []

    # iterate over batches
    total = len(train_dataloader)
    for i, batch in enumerate(train_dataloader):

        step = i+1

        # push the batch to gpu
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        del batch
        gc.collect()
        torch.cuda.empty_cache()
        # clear previously calculated gradients
        model.zero_grad()

        # get model predictions for the current batch
        #sent_id = torch.tensor(sent_id).to(device).long()
        preds = model(sent_id, mask)

        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)

        # add on to the total loss
        total_loss += float(loss.item())

        percent = "{0:.2f}".format(100 * (step / float(total)))
        lossp = "{0:.2f}".format(total_loss*100/((1+i)*batch_size))
        filledLength = int(100 * step // total)
        bar = '█' * filledLength + '>'  *(filledLength < 100) + '.' * (99 - filledLength)
        logger.info(f'\rBatch {step}/{total} |{bar}| {percent}% complete, loss={lossp}, accuracy={total_accuracy}')

        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # append the model predictions
        #total_preds.append(preds)
        total_preds.append(preds.detach().cpu().numpy())

    gc.collect()
    torch.cuda.empty_cache()

    # compute the training loss of the epoch
    avg_loss = total_loss / (len(train_dataloader)*batch_size)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    # returns the loss and predictions
    return avg_loss, total_preds

# function for evaluating the model
def evaluate():
    # deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0

    # empty list to save the model predictions
    total_preds = []

    # iterate over batches
    total = len(val_dataloader)
    for i, batch in enumerate(val_dataloader):
        
        step = i+1
        # push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch
        del batch
        gc.collect()
        torch.cuda.empty_cache()
        # deactivate autograd
        with torch.no_grad():

            # model predictions
            preds = model(sent_id, mask)

            # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds, labels)

            total_loss += float(loss.item())
            #preds = preds.detach().cpu().numpy()

            #total_preds.append(preds)
            total_preds.append(preds.detach().cpu().numpy())
        
        percent = "{0:.2f}".format(100 * (step / float(total)))
        lossp = "{0:.2f}".format(total_loss*100/((1+i)*batch_size))
        filledLength = int(100 * step // total)
        bar = '█' * filledLength + '>' * (filledLength < 100) + '.' * (99 - filledLength)
        logger.info(f'\rBatch {step}/{total} |{bar}| {percent}% complete, loss={lossp}, accuracy={total_accuracy}')

    gc.collect()
    torch.cuda.empty_cache()

    # compute the validation loss of the epoch
    avg_loss = total_loss / (len(val_dataloader)*batch_size)

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds

# Specify the GPU
# Setting up the device for GPU usage
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(device)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Load Data-set ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
train_seq, train_mask, train_y = load_and_process(os.path.join(data_dir, "train.csv"))
val_seq, val_mask, val_y = load_and_process(os.path.join(data_dir, "val.csv"))
test_seq, test_mask, test_y = load_and_process(os.path.join(data_dir, "test.csv"))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ class distribution ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# class = class label for majority of CF users. 0 - hate speech 1 - offensive language 2 - neither
# ~~~~~~~~~~~~~~~~~~~~~ Import BERT Model and BERT Tokenizer ~~~~~~~~~~~~~~~~~~~~~#
# import BERT-base pretrained model
bert = AutoModel.from_pretrained(transformer_model)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Tokenization ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# for train set
train_seq = torch.tensor(train_seq)
train_mask = torch.tensor(train_mask)
train_y = torch.tensor(train_y)

# for validation set
val_seq = torch.tensor(val_seq)
val_mask = torch.tensor(val_mask)
val_y = torch.tensor(val_y)

# for test set
test_seq = torch.tensor(test_seq)
test_mask = torch.tensor(test_mask)
test_y = torch.tensor(test_y)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Create DataLoaders ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# define a batch size
batch_size = 512

# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)

# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)

# dataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)

# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)

# dataLoader for validation set
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

# wrap tensors
test_data = TensorDataset(test_seq, test_mask, test_y)

# sampler for sampling the data during training
test_sampler = SequentialSampler(test_data)

# dataLoader for validation set
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Freeze BERT Parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# freeze all the parameters
if not ftbert:
    for param in bert.parameters():
        param.requires_grad = False
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# pass the pre-trained BERT to our define architecture
model = BERT_Arch_Simple(bert)
# push the model to GPU
model = model.to(device)

# optimizer from hugging face transformers
from transformers import AdamW

# define the optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# loss function
#cross_entropy = nn.NLLLoss(weight=weights)
cross_entropy = nn.NLLLoss()

best_valid_acc = 0

current = 1
# for each epoch
while current <= epochs:

    logger.info(f'\nEpoch {current} / {epochs}:')

    # train model
    train_loss, _ = train()
    logger.info(f'Finish epoch {current} training!')
    # evaluate model
    valid_loss, preds = evaluate()


    logger.info("Val Performance:")
    # model's performance
    preds = np.argmax(preds, axis=1)
    logger.info('Classification Report')
    logger.info(classification_report(val_y, preds))

    valid_acc = accuracy_score(val_y, preds)
    logger.info("Accuracy: " + str(valid_acc))

    # save the best model
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        torch.save(model.state_dict(), os.path.join(ckpt_dir, '{}.pth'.format(output_name)))

    logger.info(f'\n\nTraining Loss: {train_loss:.3f}')
    logger.info(f'Validation Loss: {valid_loss:.3f}')

    current = current + 1


# get predictions for test data
gc.collect()
torch.cuda.empty_cache()


with torch.no_grad():
    preds = model(test_seq.to(device), test_mask.to(device))
    preds = preds.detach().cpu().numpy()

logger.info("Test Performance:")
# model's performance
preds = np.argmax(preds, axis=1)
with open("./answer_modality_cls_output/{}.txt".format(output_name), "w") as fp:
    fp.write(json.dumps(preds.tolist()))
logger.info('Classification Report')
logger.info(classification_report(test_y, preds))

logger.info("Test Accuracy: " + str(accuracy_score(test_y, preds)))