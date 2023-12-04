import torch
from torch.utils.data import DataLoader, Dataset
import random

# BART(seq2seq)용 커스텀 데이터셋
class BartDataset(Dataset):
  def __init__(self, src_data, trg_data, now_tokenizer, mode="train"):
    self.src_dataset = src_data # list
    self.trg_dataset = trg_data # list
    self.tokenizer = now_tokenizer
    self.mode = mode

  def __len__(self):
    if self.trg_dataset != []: assert len(self.src_dataset) == len(self.trg_dataset)
    return len(self.src_dataset)

  def __getitem__(self, idx):
    src_sentence = self.src_dataset[idx]
    src_inputs = self.tokenizer(src_sentence, padding='max_length', max_length=128, truncation=True, return_tensors="pt")
    src_input_ids = torch.cat([src_inputs['input_ids'][0][1:], torch.tensor([3])]).contiguous()
    src_attention_mask = torch.cat([src_inputs['attention_mask'][0][1:], torch.tensor([0])]).contiguous()

    if self.trg_dataset != []: 
      trg_sentence = self.trg_dataset[idx]
      trg_inputs = self.tokenizer(trg_sentence, padding='max_length', max_length=128, truncation=True, return_tensors="pt")
      trg_input_ids = trg_inputs['input_ids'][0]
      trg_attention_mask = trg_inputs['attention_mask'][0]


    if self.mode == "train":
      return {
          'input_ids': src_input_ids,
          'attention_mask': src_attention_mask,
          'decoder_input_ids': trg_input_ids,
          'decoder_attention_mask': trg_attention_mask,
      }
    else: # "valid/eval"
      return {
          'input_ids': src_input_ids,
          'attention_mask': src_attention_mask,
      }
      

# Style Classifier용 커스텀 데이터셋
class StyleDataset(Dataset):
  def __init__(self, src_data, trg_data, now_tokenizer, mode="train", rand=False):
    self.total_dataset = src_data + trg_data # list
    self.total_label = [0]*len(src_data) + [1]*len(trg_data) # list
    self.tokenizer = now_tokenizer
    self.mode = mode

    if rand:
        random.seed(42)  # 랜덤 시드 고정
        random.shuffle(self.total_dataset)
        random.seed(42)  # 랜덤 시드를 같은 값으로 다시 고정하여 라벨도 셔플 순서와 일치하도록 함
        random.shuffle(self.total_label)

  def __len__(self):
    return len(self.total_dataset)

  def __getitem__(self, idx):
    sentence = self.total_dataset[idx]
    inputs = self.tokenizer(sentence, padding='max_length', max_length=128, truncation=True, return_tensors="pt")
    input_ids = inputs['input_ids'][0]
    attention_mask = inputs['attention_mask'][0]
    label = self.total_label[idx]


    if self.mode == "train":
      return {
          'input_ids': input_ids,
          'attention_mask': attention_mask,
          'label': torch.tensor(label, dtype=torch.long)
      }
    else: # "valid/eval"
      return {
          'input_ids': input_ids,
          'attention_mask': attention_mask,
      }