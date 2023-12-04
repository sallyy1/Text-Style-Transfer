import transformers
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import StyleDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score # ACC
import csv
import os
import pandas as pd
import numpy as np

device = 'cuda:0'

### Style Classifier 토크나이저 임포트 ###
sc_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
### 기학습한 Style Classifier 임포트 ###
model_saved_path = '/data/hyunkyung_lee/style_transfer/formality/outputs/formality_style_classifier_0828_20epoch'
sc_model = AutoModelForSequenceClassification.from_pretrained(model_saved_path, num_labels=2)
sc_model.to(device)
sc_model.eval()


# 파일 저장
csv_list = []
input_sentence_list= []
all_pred_list = []

# Validation 데이터셋 로드
test_0 = []

# 모델이 문체 변환한 데이터 로드
transferred_path = 'outputs/inference/greedy_formality_bart_seq2seq_0901_lr낮춤_CE_60_BLEUavg:0.8113_batch.tsv'

df = pd.read_csv(transferred_path)
# NaN 값을 빈 문자열로 교체
df['생성 문장'] = df['생성 문장'].replace({np.nan: ''})
df['타겟 문장'] = df['타겟 문장'].replace({np.nan: ''})

test_0 = df['생성 문장'].tolist()

print(test_0[:10])
print(len(test_0))


# sc_test_dataset = StyleDataset(test_0[:32], test_1[:32], sc_tokenizer, mode="eval", rand=False)
sc_test_dataset = StyleDataset(test_0, [], sc_tokenizer, mode="eval", rand=False)
sc_test_dataloader = torch.utils.data.DataLoader(sc_test_dataset, batch_size=16, shuffle=False)

# 추론 및 ACC 평가
val_losses = []
val_acc = 0
log_messages = []

all_actual_labels = torch.tensor([1] * len(test_0)) # 경상도(0) -> 표준어(1)로 바꿔야 하니까 정답 label은 모두 1

batch_index = 0
batch_size = 16

with torch.no_grad():
  for step, batch in enumerate(tqdm(sc_test_dataloader)):
      ###label = batch['label'].to(device)
      print(step)
      batch_index = step
      start_index = batch_index * batch_size
      end_index = start_index + batch_size
      batch_actual_labels = all_actual_labels[start_index:end_index]
      
      input_id= batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)

      output = sc_model(input_id, attention_mask)
      pred_logits = output.logits
      predictions = F.softmax(pred_logits, dim=1)
      predicted_labels = torch.argmax(predictions, dim=1)

      this_batch_acc = accuracy_score(predicted_labels.cpu(), batch_actual_labels.cpu())
      val_acc += this_batch_acc # accuracy_score 연산은 GPU에서 불가

      # 출력 및 파일 저장
      input_sentence_str = [sc_tokenizer.decode(sentence, skip_special_tokens=True) for sentence in input_id]
      input_sentence_list.extend(input_sentence_str) # 배치 단위임
      print("- * - * - 인풋한 문장(배치): \n" + "\n".join(s for s in input_sentence_str) + "\n")
      print(f"- * - * - 예측한 라벨(배치): {this_batch_acc}\n") # 0: 경상도 1: 표준어
      all_pred_list.extend(predicted_labels.tolist())

  result_score = round(val_acc/len(sc_test_dataloader), 4)
  print("TEST acc: ", result_score)
  print(pred_logits)
  
print(len(input_sentence_list), len(all_actual_labels.tolist()), len(all_pred_list))
csv_list = [[sentence, pred_label, answer_label, pred_label==answer_label] for sentence, pred_label, answer_label in zip(input_sentence_list, all_actual_labels.tolist(), all_pred_list)]

print(len(csv_list))
print(csv_list[0])

# csv 파일로 저장
# print(model_saved_path.split('/')[-1])
# save_path = 'outputs/inference/{0}_ACC:{1}.tsv'.format(model_saved_path.split('/')[-1], result_score)
# os.makedirs(os.path.dirname(save_path), exist_ok=True)

# with open(save_path, 'w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     writer.writerow(['인풋 문장', '예측 라벨', '정답 라벨', '맞춘 여부']) # Columns Header
#     writer.writerows(csv_list)