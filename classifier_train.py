from transformers import AutoModel, AutoTokenizer
import torch
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import StyleDataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
# random_seed = 42
# torch.manual_seed(random_seed) # DataLoader shuffle 시 랜덤시드 설정

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# BATCH_SIZE = 16
BATCH_SIZE = 32


train_0 = []

# path = 'data/경상도/total/train.0'
path = 'data/어체/train.0'

with open(path, 'r', encoding='utf-8') as file:
  for line in file:
    train_0.append(line.strip())

print(train_0[:10])


train_1 = []

# path = 'data/경상도/total/train.1'
path = 'data/어체/train.1'

with open(path, 'r', encoding='utf-8') as file2:
  for line in file2:
    train_1.append(line.strip())

print(train_1[:10])
# print(len(train_0), len(train_1))



# 모델 및  토크나이저 임포트
sc_model = AutoModelForSequenceClassification.from_pretrained("klue/bert-base", num_labels=2)
sc_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
print(sc_tokenizer)
'''
PreTrainedTokenizerFast(name_or_path='klue/bert-base', vocab_size=32000, model_max_len=512, 
is_fast=True, padding_side='right', truncation_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'})
'''
print(sc_tokenizer.tokenize('근데 그 쌤이 저한테 이렇게 말하는 거예요.'))
print(sc_tokenizer('근데 그 쌤이 저한테 이렇게 말하는 거예요.'))
print(sc_tokenizer(['근데 그 쌤이 저한테 이렇게 말하는 거예요.', '근데 그 선생님이 저한테 이렇게 말하는 거예요.'], padding=True)) # 문장 2개


'''
# 33만개 데이터 기준
sc_train_dataset = StyleDataset(train_0[:int(len(train_0)*0.8)], train_1[:int(len(train_0)*0.8)], sc_tokenizer, mode="train", rand=True)
###sc_train_dataloader = torch.utils.data.DataLoader(sc_train_dataset, batch_size=BATCH_SIZE, shuffle=False)

sc_valid_dataset = StyleDataset(train_0[int(len(train_0)*0.8):], train_1[int(len(train_0)*0.8):], sc_tokenizer, mode="train", rand=True)
###sc_valid_dataloader = torch.utils.data.DataLoader(sc_valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
'''

'''3만개 데이터 기준'''
# sc_train_dataset = StyleDataset(train_0[:24000//2], train_1[:24000//2], sc_tokenizer, mode="train", rand=True)
# sc_valid_dataset = StyleDataset(train_0[int(len(train_0)*0.8):int(len(train_0)*0.8)+6000//2], train_1[int(len(train_0)*0.8):int(len(train_0)*0.8)+6000//2], sc_tokenizer, mode="train", rand=True)

sc_train_dataset = StyleDataset(train_0[:int(len(train_0)*0.8)], train_1[:int(len(train_1)*0.8)], sc_tokenizer, mode="train", rand=True)
sc_valid_dataset = StyleDataset(train_0[int(len(train_0)*0.8):], train_1[int(len(train_1)*0.8):], sc_tokenizer, mode="train", rand=True)


len(sc_train_dataset)
print("Train Dataloader: ", len(sc_train_dataset))
print("Valid Dataloader: ", len(sc_valid_dataset))


num_train_epochs = 20
learning_rate = 2e-7
batch_size = BATCH_SIZE #128
logging_steps = len(train_0) // batch_size
# output_dir = 'total/3만_sc_trainer_0818_20epoch'
output_dir = 'formality/sc_trainer_0828_20epoch'

# Early Stopping 콜백 생성
early_stopping = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)

# TrainingArguments 설정
training_args = TrainingArguments(
                                output_dir=output_dir,
                                num_train_epochs=num_train_epochs,
                                learning_rate=learning_rate,
                                per_device_train_batch_size=batch_size,
                                per_device_eval_batch_size=batch_size,
                                
                                ###evaluation_strategy='epoch',
                                
                                logging_steps=logging_steps,
                                push_to_hub=False,
                                report_to="wandb",
                                save_strategy='epoch', # or 'epoch'
                                save_total_limit=5, # 1 -> 최종 모델만 저장됨
                                load_best_model_at_end=True,  # 최적 모델 불러오기 설정
                                evaluation_strategy='epoch',  # Early Stopping을 위해 evaluation_strategy를 steps로 설정
                                eval_steps=logging_steps,
                                dataloader_num_workers=4,
                                remove_unused_columns=False,  # 사용하지 않는 컬럼 삭제 방지
                            )

# TrainingArguments에 Early Stopping 콜백 추가
training_args.callbacks = [early_stopping]
    
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }

trainer = Trainer(model=sc_model,
                 args=training_args,
                 compute_metrics=compute_metrics,
                 train_dataset=sc_train_dataset, 
                 eval_dataset=sc_valid_dataset,
                 tokenizer=sc_tokenizer)


torch.cuda.empty_cache()

trainer.train()

# sc_model.save_pretrained('/data/hyunkyung_lee/style_transfer/saturi/outputs/total_style_classifier_0821_20epoch/')
sc_model.save_pretrained('/data/hyunkyung_lee/style_transfer/formality/outputs/formality_style_classifier_0828_20epoch/')