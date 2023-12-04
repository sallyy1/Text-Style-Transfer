import accelerate
import transformers
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
from datasets import BartDataset
from transformers import TrainingArguments
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorForSeq2Seq
from calculate import sample_3d, cal_sc_reward, cal_bl_reward, cal_reward_loss
from torch.nn import CrossEntropyLoss
import wandb
import numpy as np

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

args_subtask = 'Formality' # 'Saturi' or 'Formality'
args_wandb = True
args_search = 'greedy' # 'greedy' or 'beam'

# Wandb 초기화
if args_subtask == 'Formality':
    if args_wandb: wandb.init(reinit=True, project="Formality Transfer", entity="sally_")
elif args_subtask == 'Saturi':
    if args_wandb: wandb.init(project="Saturi Transfer", entity="sally_")

train_0 = []

if args_subtask == 'Saturi':
    path = 'data/경상도/total/train.0'
elif args_subtask == 'Formality':
    path = 'data/어체/train.0'

with open(path, 'r', encoding='utf-8') as file:
  for line in file:
    train_0.append(line.strip())
print(train_0[:10])


train_1 = []

if args_subtask == 'Saturi':
    path = 'data/경상도/total/train.1'
elif args_subtask == 'Formality':
    path = 'data/어체/train.1'

with open(path, 'r', encoding='utf-8') as file2:
  for line in file2:
    train_1.append(line.strip())
print(train_1[:10])

print(len(train_0), len(train_1))


# 모델 및  토크나이저 임포트
model_name = 'hyunwoongko/kobart'
    model_saved_path = '/data/hyunkyung_lee/style_transfer/saturi/outputs/total_bart_seq2seq_0822_CE+SC_배치문장다_mean_초기화_20' # 가중치 이어서 학습
    # model_saved_path = None # Random Initialize
    
elif args_subtask == 'Formality':
    # model_saved_path = '/data/hyunkyung_lee/style_transfer/formality/outputs/formality_bart_seq2seq_0829_CE+SC_배치문장다_mean_이어서_lr낮춤_7' # 가중치 이어서 학습
    model_saved_path = None # Random Initialize

tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name, max_length=128) # 초기화

if model_saved_path == None:
    model = BartForConditionalGeneration.from_pretrained(model_name)
else:
    model = BartForConditionalGeneration.from_pretrained(model_saved_path)
model.config.output_hidden_states = True # 설정
lm_head = model.lm_head # SC 및 BL loss 계산 시 사용 (dim=768 -> vocab_size=30000 변환하는 레이어)

# Get Data Collator
args_fp16 = 8 # False
seq2seq_datacollator = DataCollatorForSeq2Seq(tokenizer, model,
                                                label_pad_token_id=tokenizer.pad_token_id,
                                                pad_to_multiple_of=8 if args_fp16 else None)

# 데이터셋 로드
random_seed = 42
torch.manual_seed(random_seed) # DataLoader shuffle 시 랜덤시드 설정

args_batch_size = 16 #32

train_dataset = BartDataset(train_0[:int(len(train_0)*0.8)], train_1[:int(len(train_0)*0.8)], tokenizer, mode="train")
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args_batch_size, shuffle=True, collate_fn=seq2seq_datacollator)

valid_dataset = BartDataset(train_0[int(len(train_0)*0.8):], train_1[int(len(train_0)*0.8):], tokenizer, mode="train")
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args_batch_size, shuffle=True, collate_fn=seq2seq_datacollator)


len(train_dataset)
print("Train Dataloader: ", len(train_dataloader))
print("Valid Dataloader: ", len(valid_dataloader))

logging_steps = len(train_dataset) // args_batch_size
args_num_train_epochs = 60 #30
args_savemodel_units = 5 # or False
args_learning_rate = 3e-5 # 2e-7 # 3e-6
torch.cuda.set_device(device)
torch.cuda.empty_cache()

### Style Classifier 토크나이저 임포트 ###
sc_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
### 기학습한 Style Classifier 임포트 ###
if args_subtask == 'Saturi':
    sc_model = AutoModelForSequenceClassification.from_pretrained('/data/hyunkyung_lee/style_transfer/saturi/outputs/total_style_classifier_0821_20epoch/', num_labels=2)
elif args_subtask == 'Formality':
    sc_model = AutoModelForSequenceClassification.from_pretrained('/data/hyunkyung_lee/style_transfer/formality/outputs/formality_style_classifier_0828_20epoch/', num_labels=2)
sc_model.to(device)
sc_model.eval()

# calculate(transferred_sentence_str, sc_model, sc_tokenizer)
# batch_target_reward = cal_sc_reward(transferred_sentence_str, sc_model, sc_tokenizer, style_label=0) # 경상도(0) -> 표준어(1) 변환 시

### train 코드 ###
optimizer = Adam(model.parameters(), lr=args_learning_rate)
model.to(device) ### 모델도 이동시켜야 함
torch.cuda.set_device(device)
torch.cuda.empty_cache()

loss_fn = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# 시작 토큰, 종료 토큰, 패딩 토큰 확인
start_token_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
end_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
padding_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

print("Start Token ID:", start_token_id) # 0
print("End Token ID:", end_token_id) # 1
print("Padding Token ID:", padding_token_id) # 3

def train_epoch(model, optimizer, train_dataloader, SC_LOSS=False, BL_LOSS=False):
    model.train()
    train_loss = 0.0

    CE_loss = []
    SC_loss = []
    BL_loss = []

    for step, batch in enumerate(tqdm(train_dataloader)):
        print(len(batch))
        input_ids = batch['input_ids'].to(device) ### 데이터(텐서) 이동시킴
        attention_mask = batch['attention_mask'].to(device)
        trg_input_ids = batch['decoder_input_ids'].to(device)
        trg_attention_mask = batch['decoder_attention_mask'].to(device)
        
        optimizer.zero_grad()

        # Forward pass
        # 학습 시
        # (1) Cross Entropy loss
        labels = trg_input_ids.clone().detach() # Tensor 복사        
        # new_tensor = torch.full([args_batch_size, 1], 3).to(device)
        new_tensor = torch.full_like(labels[:, :1], 3).to(device)
        labels = torch.cat((labels[..., 1:], new_tensor), dim=-1).contiguous()
        
        BART_outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=trg_input_ids, labels=labels) # Teacher Forcing
        
        ce_loss = BART_outputs.loss # CrossEntropy Loss ### (1) 번 시도
        ######logits = BART_outputs[0] ### 디버깅 중 ### 
        '''
        ce_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1)) ### (2) 번 시도
        '''
        
        
        '''
        # print("* = * = * = * = *: ", BART_outputs.logits)
        # print("* = * = * = * = *: ", BART_outputs[0])
        # print("* = * = * = * = *: ", BART_outputs.loss)
        logits = BART_outputs.logits # (batch_size, sequence_length, config.vocab_size)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce_loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), ### (3) 번 시도
                          shift_labels.view(-1))
        '''
        
        
        ''' Teacher Forcing
        - 'decoder_input_ids': 모델 입력으로 사용할 디코더의 입력 시퀀스 인풋
        - 'labels': 모델이 생성한 시퀀스와 비교할 정답 시퀀스 인풋
        '''
        print("C.E loss: ", ce_loss)
        CE_loss.append(ce_loss.item())
        
        
        
        # print("= input_ids: ", input_ids[0])
        # print("= trg_input_ids: ", trg_input_ids[0])
        # print("= input_ids: ", input_ids.size()) # torch.Size([16, 128])
        # print("= trg_input_ids: ", trg_input_ids.size()) # torch.Size([16, 128])
        
        
        
        sc_loss, bl_loss = torch.tensor(0), torch.tensor(0) # 기본 값은 0        
        # 추가 Loss
        if SC_LOSS or BL_LOSS: 
            '''BART_output: transferred_sentence'''
            hidden_states = BART_outputs.decoder_hidden_states
            last_hidden_state = hidden_states[-1] # 최종 생성된 문장의 hidden states: (batch_size, sequence_length, config.d_model)
            #print("last_hidden_state 결과: ", last_hidden_state.size())
            
            ###out = F.softmax(last_hidden_state.float(), dim=-1) # torch.Size([16, 20]) = (batch_size, seq_len) ### 확인 필요 !!!
            
            # dim=768 -> vocab idx 변환
            vocab_idx_outputs = lm_head(last_hidden_state)
            out = F.softmax(vocab_idx_outputs.float(), dim=-1)
            
            #print("= last_hidden_state의 probs 결과: ", last_hidden_state.size()) # torch.Size([16, 128, 768])
            #print(last_hidden_state)
            #print("= vocab_idx_outputs 결과: ", vocab_idx_outputs.size()) # torch.Size([16, 128, 30000])
            #print(vocab_idx_outputs)
            sample_probs, sample_idx = sample_3d(out) # sample_3d의 인풋 텐서는 (batch, seq_len, vocab_size) 여야 함
            # 샘플링한 확률 및 인덱스: torch.Size([16, 128])
            #print("= sample_probs 결과: ", sample_probs.size(), sample_idx.size()) # 둘다 torch.Size([16, 128]) = (batch_size, max_seq_len, 1)
            #print(sample_probs)
            #print(sample_idx)
            
            greedy_probs, greedy_idx = torch.max(out, dim=-1) # 그리디한 확률 및 인덱스: torch.Size([16, 128])
            #print("= greedy_probs 결과: ", greedy_probs.size(), greedy_idx.size())
            #print(greedy_probs)
            #print(greedy_idx)
            
            #trg_input_ids = trg_input_ids.squeeze(-1) # torch.Size([16, 128, 768]) -> torch.Size([16, 128])
            #print(trg_input_ids.size())
            

            # -> 디코딩된 문장 변환
            sampling_decoded_text = tokenizer.batch_decode(sample_idx, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            ###print(sampling_decoded_text) # [16]
            greedy_decoded_text = tokenizer.batch_decode(greedy_idx, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            ###print(greedy_decoded_text) # [16]
            target_decoded_text = tokenizer.batch_decode(trg_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            ###print(target_decoded_text.size()) # List (len: 16)
        
        if SC_LOSS:
            # (2) Style Classifier Reward based loss
            batch_target_reward = cal_sc_reward(sampling_decoded_text, sc_model, sc_tokenizer, style_label=0) # 경상도(0) -> 표준어(1) 변환 시 # 합쇼체(0) -> 반말체(1) 변환 시
            
            # 배치 0번에 대해서만 loss 구할 때
            '''
            target_reward = torch.tensor(batch_target_reward[0]) # List to Tensor
            ###print("target_reward 값: ", target_reward, target_reward.size())
            sc_loss = cal_reward_loss(sample_probs, reward=target_reward.to(device), idxs=None)
            ###print("S.C loss: ", sc_loss, sc_loss.size()) # S.C loss:  tensor(-0.0820, device=args_device, grad_fn=<MeanBackward0>) torch.Size([])
            print("S.C loss: ", sc_loss)
            SC_loss.append(sc_loss.item())
            '''
            
            
            # 저자의 코드에서는 각 배치에서 0번째 시퀀스에 대해서만 S.C loss 및 B.L loss를 계산하였음
            # 배치 전체에 대한 loss를 구하려면?
            batch_sc_loss = []
            for idx in range(len(batch)): # args_batch_size
                target_reward = torch.tensor(batch_target_reward[idx]) # List to Tensor
                sc_loss = cal_reward_loss(sample_probs, reward=target_reward.to(device), idxs=None)
                batch_sc_loss.append(sc_loss)
            sc_loss = torch.stack(batch_sc_loss).mean() # 1차원 텐서로 쌓아 평균값 계산
            
            print("[Batch mean] S.C loss: ", sc_loss)
            SC_loss.append(sc_loss.item())
            
            
        if BL_LOSS:    
            # (3) Content Preservation Reward based loss
            batch_target_sampling = cal_bl_reward(sampling_decoded_text, target_decoded_text, device)
            target_sampling = batch_target_sampling[0]
            batch_target_greedy = cal_bl_reward(greedy_decoded_text, target_decoded_text, device)
            target_greedy = batch_target_greedy[0]
            
            '''
            bleu_reward = ((target_greedy - target_sampling) * 0.2).unsqueeze(0) # 차원이 없는 Scalar 값 -> 1차원 변환
            ###print(bleu_reward)

            print('= = = = = = = = = = = =')
            print("sampling_decoded_text 값: ", sampling_decoded_text[0])
            print("greedy_decoded_text 값: ", greedy_decoded_text[0])
            print("target_decoded_text 값: ", target_decoded_text[0])
            print(batch_target_sampling, batch_target_sampling.size())
            ###print("bleu_reward 값: ", bleu_reward.size())
            
            bl_loss = cal_reward_loss(sample_probs, reward=bleu_reward.to(device), idxs=None)
            #print("B.L reward: ", bleu_reward)
            print("B.L loss: ", bl_loss)
            BL_loss.append(bl_loss.item())
            '''
            
            # 저자의 코드에서는 각 배치에서 0번째 시퀀스에 대해서만 S.C loss 및 B.L loss를 계산하였음
            # 배치 전체에 대한 loss를 구하려면?
            batch_bleu_reward = ((batch_target_greedy - batch_target_sampling) * 0.2).unsqueeze(-1)
            #print(batch_bleu_reward)
            #print("batch_bleu_reward: ", batch_bleu_reward.size())
            #print("batch_bleu_reward[0]: ", batch_bleu_reward[0].size())
                    
            batch_bl_loss = []
            for idx in range(len(batch)): # args_batch_size
                bl_loss = cal_reward_loss(sample_probs, reward=batch_bleu_reward[idx].to(device), idxs=None)
                batch_bl_loss.append(bl_loss)
            bl_loss = torch.stack(batch_bl_loss).mean() # 1차원 텐서로 쌓아 평균값 계산
            print("[Batch mean] B.L loss: ", bl_loss)
            
            # bl_loss = torch.stack(batch_bl_loss).sum() # 1차원 텐서로 쌓아 합 계산
            # print("[Batch sum] B.L loss: ", bl_loss)
            BL_loss.append(bl_loss.item())
            
        
        train_loss = (ce_loss + sc_loss + bl_loss)

        # Backward pass and optimization
        train_loss.backward()
        optimizer.step()

    average_train_loss = (sum(CE_loss)+sum(SC_loss)+sum(BL_loss)) / len(train_dataloader)
    return average_train_loss, CE_loss, SC_loss, BL_loss


def eval_epoch(model, valid_dataloader, SC_LOSS=False, BL_LOSS=False):
    model.eval()
    valid_loss = 0.0

    valid_CE_loss = []
    valid_SC_loss = []
    valid_BL_loss = []
    
    with torch.no_grad():
        for val_batch in valid_dataloader:
            input_ids = val_batch['input_ids'].to(device) ### 데이터(텐서) 이동시킴
            attention_mask = val_batch['attention_mask'].to(device)
            trg_input_ids = val_batch['decoder_input_ids'].to(device)
            trg_attention_mask = val_batch['decoder_attention_mask'].to(device)
            
            # 추론 시
            # (1) Cross Entropy loss
            labels = trg_input_ids.clone().detach() # Tensor 복사        
            # new_tensor = torch.full([args_batch_size, 1], 3).to(device)
            new_tensor = torch.full_like(labels[:, :1], 3).to(device)
            labels = torch.cat((labels[..., 1:], new_tensor), dim=-1).contiguous()
            
            BART_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels) # Model Forward (Teacher Forcing X)
            
            ce_loss = BART_outputs.loss # CrossEntropy Loss ### (1) 번 시도
            ######logits = BART_outputs[0] ### 디버깅 중 ### 
            '''
            ce_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1)) ### (2) 번 시도
            '''
            
            '''
            # print("* = * = * = * = *: ", BART_outputs.logits)
            # print("* = * = * = * = *: ", BART_outputs[0])
            # print("* = * = * = * = *: ", BART_outputs.loss)
            logits = BART_outputs.logits # (batch_size, sequence_length, config.vocab_size)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            '''       
            
            print("C.E loss(Valid): ", ce_loss)
            valid_CE_loss.append(ce_loss.item())
    
            sc_loss, bl_loss = torch.tensor(0), torch.tensor(0) # 기본 값은 0
            # 추가 Loss
            if SC_LOSS or BL_LOSS: 
                '''BART_output: transferred_sentence'''
                hidden_states = BART_outputs.decoder_hidden_states
                last_hidden_state = hidden_states[-1] # 최종 생성된 문장의 hidden states: (batch_size, sequence_length, config.d_model)

                # dim=768 -> vocab idx 변환
                vocab_idx_outputs = lm_head(last_hidden_state)
                out = F.softmax(vocab_idx_outputs.float(), dim=-1)
                
                sample_probs, sample_idx = sample_3d(out) # sample_3d의 인풋 텐서는 (batch, seq_len, vocab_size) 여야 함
                # 샘플링한 확률 및 인덱스: torch.Size([16, 128])
                greedy_probs, greedy_idx = torch.max(out, dim=-1) # 그리디한 확률 및 인덱스: torch.Size([16, 128])
                

                # -> 디코딩된 문장 변환
                sampling_decoded_text = tokenizer.batch_decode(sample_idx, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                ###print(sampling_decoded_text) # [16]
                greedy_decoded_text = tokenizer.batch_decode(greedy_idx, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                ###print(greedy_decoded_text) # [16]
                target_decoded_text = tokenizer.batch_decode(trg_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                ###print(target_decoded_text.size()) # List (len: 16)
            
            if SC_LOSS:
                # (2) Style Classifier Reward based loss
                batch_target_reward = cal_sc_reward(sampling_decoded_text, sc_model, sc_tokenizer, style_label=0) # 경상도(0) -> 표준어(1) 변환 시 
                
                # 배치 0번에 대해서만 loss 구할 때
                '''
                target_reward = torch.tensor(batch_target_reward[0]) # List to Tensor
                ###print("target_reward 값: ", target_reward, target_reward.size())
                sc_loss = cal_reward_loss(sample_probs, reward=target_reward.to(device), idxs=None)
                ###print("S.C loss: ", sc_loss, sc_loss.size()) # S.C loss:  tensor(-0.0820, device=args_device, grad_fn=<MeanBackward0>) torch.Size([])
                print("S.C loss(Valid): ", sc_loss)
                valid_SC_loss.append(sc_loss.item())
                '''
                
                # 저자의 코드에서는 각 배치에서 0번째 시퀀스에 대해서만 S.C loss 및 B.L loss를 계산하였음
                # 배치 전체에 대한 loss를 구하려면?
                batch_sc_loss = []
                for idx in range(len(val_batch)): # args_batch_size
                    target_reward = torch.tensor(batch_target_reward[idx]) # List to Tensor
                    sc_loss = cal_reward_loss(sample_probs, reward=target_reward.to(device), idxs=None)
                    batch_sc_loss.append(sc_loss)
                sc_loss = torch.stack(batch_sc_loss).mean() # 1차원 텐서로 쌓아 평균값 계산
                
                print("[Batch mean] S.C loss(Valid): ", sc_loss)
                valid_SC_loss.append(sc_loss.item())
                
                
                
                
            if BL_LOSS:    
                # (3) Content Preservation Reward based loss
                batch_target_sampling = cal_bl_reward(sampling_decoded_text, target_decoded_text, device)
                target_sampling = batch_target_sampling[0]
                batch_target_greedy = cal_bl_reward(greedy_decoded_text, target_decoded_text, device)
                target_greedy = batch_target_greedy[0]
                
                '''
                bleu_reward = ((target_greedy - target_sampling) * 0.2).unsqueeze(0) # 차원이 없는 Scalar 값 -> 1차원 변환
                
                print('= = = = = = = = = = = =')
                print("sampling_decoded_text 값: ", sampling_decoded_text[0])
                print("greedy_decoded_text 값: ", greedy_decoded_text[0])
                print("target_decoded_text 값: ", target_decoded_text[0])
                print(batch_target_sampling, batch_target_sampling.size())
                ###print("bleu_reward 값: ", bleu_reward.size())
                
                bl_loss = cal_reward_loss(sample_probs, reward=bleu_reward.to(device), idxs=None)
                print("B.L loss(Valid): ", bl_loss)
                valid_BL_loss.append(bl_loss.item())
                '''
                
                
                # 저자의 코드에서는 각 배치에서 0번째 시퀀스에 대해서만 S.C loss 및 B.L loss를 계산하였음
                # 배치 전체에 대한 loss를 구하려면?
                batch_bleu_reward = ((batch_target_greedy - batch_target_sampling) * 0.2).unsqueeze(-1)
                print(batch_bleu_reward)
                print(batch_bleu_reward[0].size())
                        
                batch_bl_loss = []
                for idx in range(len(val_batch)): # args_batch_size
                    bl_loss = cal_reward_loss(sample_probs, reward=batch_bleu_reward[idx].to(device), idxs=None)
                    batch_bl_loss.append(bl_loss)
                bl_loss = torch.stack(batch_bl_loss).mean() # 1차원 텐서로 쌓아 평균값 계산
                print("[Batch mean] B.L loss(Valid): ", bl_loss)
                
                # bl_loss = torch.stack(batch_bl_loss).sum() # 1차원 텐서로 쌓아 합 계산
                # print("[Batch mean] B.L loss(Valid): ", bl_loss)
                valid_BL_loss.append(bl_loss.item())
                
            
            valid_loss = (ce_loss + sc_loss + bl_loss)

            # Convert generated sentences back to strings
            if args_search == 'greedy':
                transferred_sentence = model.generate(input_ids, attention_mask=attention_mask)
                transferred_sentence_str = [tokenizer.decode(sentence, skip_special_tokens=True) for sentence in transferred_sentence]
            elif args_search == 'beam':
                transferred_sentence = model.generate(input_ids, attention_mask=attention_mask, num_beam=5, early_stopping=True)
                # transferred_sentence = model.generate(input_ids, attention_mask=attention_mask, num_beam=5, early_stopping=True, no_repeat_ngram_size=2) # 2-gram의 어구가 반복되지 않도록 설정함
                transferred_sentence_str = [tokenizer.decode(sentence, skip_special_tokens=True) for sentence in transferred_sentence]                
            
            input_sentence_str = [tokenizer.decode(sentence, skip_special_tokens=True) for sentence in input_ids]
            print("- * - * - 인풋한 문장(배치): \n" + "\n".join(s for s in input_sentence_str) + "\n")
            print("- * - * - 변환한 문장(배치): \n" + "\n".join(s for s in transferred_sentence_str))

        average_valid_loss = (sum(valid_CE_loss)+sum(valid_SC_loss)+sum(valid_BL_loss)) / len(valid_dataloader)          
    
    return average_valid_loss, valid_CE_loss, valid_SC_loss, valid_BL_loss

# Training loop
args_sc = False
args_bl = False


# 하이퍼파라미터 설정
config = {
    "num_train_epochs": args_num_train_epochs,
    "learning_rate": args_learning_rate,
    "batch_size": args_batch_size,
    "args_sc": args_sc,
    "args_bl": args_bl,
}

# wandb.config에 설정 저장
if args_wandb: wandb.config.update(config)

# Initialize variables for early stopping
best_valid_loss = np.inf
patience = 3
no_improvement_count = 0

for epoch in range(args_num_train_epochs):
    print(f"Epoch {epoch + 1}/{args_num_train_epochs}")
    train_loss, CE_loss, SC_loss, BL_loss = train_epoch(model, optimizer, train_dataloader, SC_LOSS=args_sc, BL_LOSS=args_bl)
    epoch_CE_loss = sum(CE_loss)/len(train_dataloader)
    epoch_SC_loss = sum(SC_loss)/len(train_dataloader)
    epoch_BL_loss = sum(BL_loss)/len(train_dataloader)
    
    print(f"Train Loss: {train_loss:.4f}")
    print(f"- Cross Entropy loss epoch avg: {epoch_CE_loss: 4f}", len(CE_loss))
    if args_sc: print(f"- Style loss epoch avg: {epoch_SC_loss: 4f}", len(SC_loss))
    if args_bl: print(f"- BLEU loss epoch avg: {epoch_BL_loss: 4f}", len(BL_loss))
    
    valid_loss, valid_CE_loss, valid_SC_loss, valid_BL_loss = eval_epoch(model, valid_dataloader, SC_LOSS=args_sc, BL_LOSS=args_bl)
    valid_epoch_CE_loss = sum(valid_CE_loss)/len(valid_dataloader)
    valid_epoch_SC_loss = sum(valid_SC_loss)/len(valid_dataloader)
    valid_epoch_BL_loss = sum(valid_BL_loss)/len(valid_dataloader)
    
    print(f"Valid Loss: {valid_loss:.4f}")
    print(f"- Cross Entropy loss epoch avg(Valid): {valid_epoch_CE_loss: 4f}", len(valid_CE_loss))
    if args_sc: print(f"- Style loss epoch avg(Valid): {valid_epoch_SC_loss: 4f}", len(valid_SC_loss))
    if args_bl: print(f"- BLEU loss epoch avg(Valid): {valid_epoch_BL_loss: 4f}", len(valid_BL_loss))    


    # Check if the validation loss improved
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        no_improvement_count = 0
        # Save the model if needed
        # if args_savemodel_units and (epoch + 1) % args_savemodel_units == 0:
        # if (epoch + 1) >= args_savemodel_units:
        if (epoch + 1):
            if args_subtask == 'Saturi':
                model.save_pretrained(f'/data/hyunkyung_lee/style_transfer/saturi/outputs/total_bart_seq2seq_0831_CE+SC+BL_배치문장다_mean_2번에이어서_patience3_{15+ epoch + 1}')
            elif args_subtask == 'Formality':
                model.save_pretrained(f'/data/hyunkyung_lee/style_transfer/formality/outputs/formality_bart_seq2seq_0901_3e-6_CE_{epoch + 1}')
    else:
        no_improvement_count += 1
        if no_improvement_count >= patience:
            print("No improvement for {} epochs. Early stopping.".format(patience))
            break


    # Wandb 로그 기록
    if args_wandb:
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_CE_loss": epoch_CE_loss,
            "train_SC_loss": epoch_SC_loss if args_sc else None,
            "train_BL_loss": epoch_BL_loss if args_bl else None,
            "valid_loss": valid_loss,
            "valid_CE_loss": valid_epoch_CE_loss,
            "valid_SC_loss": valid_epoch_SC_loss if args_sc else None,
            "valid_BL_loss": valid_epoch_BL_loss if args_bl else None,
            "check_earlystop_patience": no_improvement_count,
        })
        
    # {args_savemodel_units} epoch마다 모델 저장
    # if args_savemodel_units and (epoch + 1) % args_savemodel_units == 0:        
    #     model.save_pretrained(f'/data/hyunkyung_lee/style_transfer/formality/outputs/formality_bart_seq2seq_0828_CE_{epoch + 1}')
          

# if args_savemodel_units == False:
#     model.save_pretrained('/data/hyunkyung_lee/style_transfer/saturi/outputs/total_bart_seq2seq_0817_CE+BL_배치문장다_sum_초기화_')