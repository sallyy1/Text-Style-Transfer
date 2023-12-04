import transformers
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import BartDataset
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast, BartConfig
from calculate import cal_bl_reward
import os
import csv

args_subtask = 'Formality' # 'Saturi' or 'Formality'
args_search = 'greedy' # 'greedy' of 'beam'
args_infer_fashion = 'batch' # 'each' or 'batch'
args_batch_size = 16
model_path = '/data/hyunkyung_lee/style_transfer/formality/outputs/formality_bart_seq2seq_0901_3e-6_CE_22'

model = BartForConditionalGeneration.from_pretrained(model_path)
tokenizer = PreTrainedTokenizerFast.from_pretrained('hyunwoongko/kobart', max_length=128)

# print(model)
# 시작 토큰, 종료 토큰, 패딩 토큰 확인
start_token_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
end_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
padding_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

print("Start Token ID:", start_token_id) # 0
print("End Token ID:", end_token_id) # 1
print("Padding Token ID:", padding_token_id) # 3


device = 'cuda:0' # CPU
model.to(device)
model.eval()


# Validation 데이터셋 로드
test_0 = []

if args_subtask == 'Saturi':
    path = 'data/경상도/total/test.0'
elif args_subtask == 'Formality':
    path = 'data/어체/test.0'

with open(path, 'r', encoding='utf-8') as file:
  for line in file:
    test_0.append(line.strip())

print(test_0[:10])


test_1 = []

if args_subtask == 'Saturi':
    path = 'data/경상도/total/test.1' # 정답 데이터
elif args_subtask == 'Formality':
    path = 'data/어체/test.1' # 정답 데이터

with open(path, 'r', encoding='utf-8') as file2:
  for line in file2:
    test_1.append(line.strip())

print(test_1[:10])

print(len(test_0), len(test_1))

# 파일 저장
csv_list = []
genpair_list = []


test_dataset = BartDataset(test_0, [], tokenizer, mode="eval")
# test_dataset = BartDataset(test_0[:32], test_1[:32], tokenizer, mode="train")
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args_batch_size, shuffle=False)

if args_infer_fashion == 'batch':
  with torch.no_grad():
      for step, batch in enumerate(tqdm(test_dataloader)):
          input_ids = batch['input_ids'].to(device) ### 데이터(텐서) 이동시킴
          attention_mask = batch['attention_mask'].to(device)
          
          if args_search == 'greedy':
              transferred_sentence = model.generate(input_ids, attention_mask=attention_mask, max_length=128)
              transferred_sentence_str = [tokenizer.decode(sentence, skip_special_tokens=True) for sentence in transferred_sentence]
          elif args_search == 'beam':
              transferred_sentence = model.generate(input_ids, attention_mask=attention_mask, num_beams=5, early_stopping=True, max_length=128)
              # transferred_sentence = model.generate(input_ids, attention_mask=attention_mask, num_beam=5, early_stopping=True, no_repeat_ngram_size=2, max_length=128) # 2-gram의 어구가 반복되지 않도록 설정함
              transferred_sentence_str = [tokenizer.decode(sentence, skip_special_tokens=True) for sentence in transferred_sentence]  
              
                   
          print("인풋한 텐서: ", input_ids[0])
          # print("디코더 인풋 텐서: ", trg_input_ids[0])
          print("생성한 텐서: ", transferred_sentence[0])
          transferred_sentence_str = [tokenizer.decode(sentence, skip_special_tokens=True) for sentence in transferred_sentence]
          
          input_sentence_str = [tokenizer.decode(sentence, skip_special_tokens=True) for sentence in input_ids]
          print("- * - * - 인풋한 문장(배치): \n" + "\n".join(s for s in input_sentence_str) + "\n")
          print("- * - * - 변환한 문장(배치): \n" + "\n".join(s for s in transferred_sentence_str))
          genpair_list.extend([x, y] for x, y in zip(input_sentence_str, transferred_sentence_str))


else:
  with torch.no_grad():
      for step, batch in enumerate(tqdm(test_dataloader)):
          input_ids = batch['input_ids'].to(device) ### 데이터(텐서) 이동시킴
          attention_mask = batch['attention_mask'].to(device)
          
          transferred_sentence_str = []
          
          for idx in range(args_batch_size):
              input_sentence_str = tokenizer.decode(input_ids[idx], skip_special_tokens=True)
              print("- * - * - 인풋한 문장(각 문장): \n" + input_sentence_str + "\n")
              
              transferred_sentence = model.generate(input_ids[idx].unsqueeze(dim=0), attention_mask=attention_mask[idx].unsqueeze(dim=0), max_length=128)
              transferred_sentence_str = tokenizer.batch_decode(transferred_sentence, skip_special_tokens=True)[0]
              print("= = = = = Greedy Search = = = = =")
              print("- * - * - 변환한 문장(각 문장): \n" + transferred_sentence_str + "\n")
              genpair_list.append([input_sentence_str, transferred_sentence_str])

              
              transferred_sentence = model.generate(input_ids[idx].unsqueeze(dim=0), attention_mask=attention_mask[idx].unsqueeze(dim=0), num_beams=5, early_stopping=True, max_length=128)
              # transferred_sentence = model.generate(input_ids[idx].unsqueeze(dim=0), attention_mask=attention_mask[idx].unsqueeze(dim=0), num_beam=5, early_stopping=True, max_length=128, no_repeat_ngram_size=2) # 2-gram의 어구가 반복되지 않도록 설정함
              transferred_sentence_str = tokenizer.batch_decode(transferred_sentence, skip_special_tokens=True)[0]
              print("= = = = = Beam Search = = = = =")
              # print("- * - * - 인풋한 문장(각 문장): \n" + input_sentence_str + "\n")
              print("- * - * - 변환한 문장(각 문장): \n" + transferred_sentence_str + "\n")
              print()
              print()
              genpair_list.append([input_sentence_str, transferred_sentence_str])

test_bleu = cal_bl_reward([generated_data[-1] for generated_data in genpair_list], test_1, device).tolist()
result_score = round(sum(test_bleu) / len(test_dataset), 4)
print("Total average BLEU Score: ", result_score)

print(len(genpair_list))
print(genpair_list[0])

csv_list = [[*generated_data, answer, round(bleu_score, 4)] for generated_data, answer, bleu_score in zip(genpair_list, test_1, test_bleu)]

print(len(csv_list))
print(csv_list[0])

# csv 파일로 저장
save_path = 'outputs/inference/{3}_{0}_BLEUavg:{1}_{2}.tsv'.format(model_path.split('/')[-1], result_score, args_infer_fashion, args_search)
os.makedirs(os.path.dirname(save_path), exist_ok=True)

with open(save_path, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['소스 문장', '생성 문장', '타겟 문장', 'BLEU 스코어']) # Columns Header
    writer.writerows(csv_list)
