'''
Takes raw text and saves BERT-cased features for that text to disk

Adapted from the BERT readme (and using the corresponding package) at

https://github.com/huggingface/pytorch-pretrained-BERT

###
John Hewitt, johnhew@stanford.edu
Feb 2019

'''
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, WordpieceTokenizer
from argparse import ArgumentParser
import h5py
import numpy as np

argp = ArgumentParser()
argp.add_argument('input_path')
argp.add_argument('output_path')
argp.add_argument('bert_model')
args = argp.parse_args()

# Load pre-trained model tokenizer (vocabulary)
# Crucially, do not do basic tokenization; PTB is tokenized. Just do wordpiece tokenization.
tokenizer = BertTokenizer.from_pretrained(args.bert_model)
model = BertModel.from_pretrained(args.bert_model)
LAYER_COUNT = model.config.num_hidden_layers
FEATURE_COUNT = model.config.hidden_size

model.eval()

with h5py.File(args.output_path, 'w') as fout:
  for index, line in enumerate(open(args.input_path)):
    line = line.strip() # Remove trailing characters
    line = '[CLS] ' + line + ' [SEP]'
    tokenized_text = tokenizer.wordpiece_tokenizer.tokenize(line)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segment_ids = [1 for x in tokenized_text]
  
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segment_ids])
  
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)
    dset = fout.create_dataset(str(index), (LAYER_COUNT, len(tokenized_text), FEATURE_COUNT))
    dset[:,:,:] = np.vstack([np.array(x) for x in encoded_layers])
  

