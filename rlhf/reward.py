import json
import os

import torch
from beartype import beartype
# from beartype.typing import Optional, Iterable
from typing import Optional, Iterable, List
from einops.layers.torch import Rearrange
###from langchain import OpenAI, LLMChain, PromptTemplate # Reward 모델로 생성 모델 안 씀
from torch.utils.data import Dataset, DataLoader
# from transformers import GPT2Tokenizer, GPT2Model, BartModel # Reward 모델로 생성 모델 안 씀
# from transformers import BartTokenizer, BartConfig, AutoModel, AutoTokenizer # Reward 모델로 생성 모델 안 씀
from transformers import AutoModelForSequenceClassification, AutoTokenizer#, DataCollatorForSeq2Seq
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
# from datasets import StyleDataset
import random

###from chatllama.langchain_modules.prompt_templates import REWARD_TEMPLATE # Reward 모델로 생성 모델 안 씀
from config import ConfigReward
from utils import TrainingStats

# from calculate import cal_sc_reward
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }

class RewardModel(torch.nn.Module):
    """Model to be trained to predict the reward for RL.
    or to be used as Critic in RL.

    Attributes:
        model (torch.nn.Module): Model to be used for the reward model
        tokenizer (torch.nn.Module): Tokenizer to be used for the reward model
        head (torch.nn.Module): Head to be used for the reward model
        config (ConfigReward): Config parameters for the reward model
        max_model_tokens (int): Maximum sequence length for the reward model

    Methods:
        forward: Forward pass of the model (used by the critic)
        save: Save the model
        load: Load the model
        get_reward: Get the reward for a given input (used by the reward model)
    """

    def __init__(self, config: ConfigReward) -> None:
        super().__init__()
        # load the model -- add here other models
        # store config
        self.config = config
        if os.path.exists(config.model_folder) is False:
            os.mkdir(config.model_folder)
        else:
            self.load()

        # move model to device
        self.model.to(config.device)

    @beartype
    def parameters(
        self,
    ) -> Iterable[torch.nn.Parameter]:
        """Return the parameters of the reward model"""
        for p in self.model.parameters():
            yield p

    @beartype
    def forward(
        self, transferred_sentence_str: List, style_label: int
    ) -> torch.Tensor:
        """Generate the sequence of rewards for the given output sequence
        what is the quality of the output sequence tokens?

        Args:
            output_sequence (torch.Tensor): The sequence of tokens to be
                evaluated
            output_sequence_mask (torch.Tensor): Mask for the attention

        Returns:
            torch.Tensor: Rewards for the given output sequence
        """
         
        # Rewards 값 계산
        # (1) Style Classifier Reward based loss
        
        # BERT(Style Classifier) 토크나이저로 토큰화
        # str -> tensor
        # calculate.py의 cal_sc_reward 함수
        batch_target_reward = []
        
        for trans_sen in transferred_sentence_str:
            trans_inputs = self.sc_tokenizer(trans_sen, padding=True, return_tensors="pt") # return_tensors="pt"
            input_id = trans_inputs['input_ids'].to(self.config.device)
            attention_mask = trans_inputs['attention_mask'].to(self.config.device)        
            
            # output = self.model(output_sequence, attention_mask=output_sequence_mask)
            output = self.model(input_id, attention_mask=attention_mask)
            pred_logits = output.logits
            predictions = F.softmax(pred_logits, dim=1) # tensor([[0.7079, 0.2921]])

            if style_label==0: # 경상도(0) -> 표준어(1) NO // 표준어(1) -> 경상도(0) YES!!
                target_reward = (predictions[:, 1] - predictions[:, 0]) # <class 'torch.Tensor'>, torch.Size([1])
            elif style_label==1: # 표준어(1) -> 경상도(0) // 경상도(0) -> 표준어(1) YES!!
                target_reward = (predictions[:, 0] - predictions[:, 1]) # <class 'torch.Tensor'>, torch.Size([1])

            batch_target_reward.append(target_reward) # Scalar 값을 List에 추가
            ###print("STYLE target reward 값: ", type(target_reward)) # torch.Size([1])
            ###print(target_reward.size())

        # List -> torch 변환
        rewards = torch.tensor(batch_target_reward)
        
        
        if self.config.debug:
            print("RewardModel.forward")
            # print("output_sequence.shape", output_sequence.shape)
            # print("output_sequence", output_sequence)
            print("output_sequence.shape", transferred_sentence_str.shape)
            print("output_sequence", transferred_sentence_str)
            print("reward.shape", rewards.shape)
            print("reward", rewards)
        return rewards # 배치 단위

    @beartype
    def get_reward(
        self, transferred_sentence_str: List, style_label: int
    ) -> torch.Tensor:
        """Get the reward for the given output sequence

        Args:
            output_sequence (torch.Tensor): The concatenation of initial input
                and actor output as tokens
            output_sequence_mask (torch.Tensor): Mask for the attention
        """
        rewards = self.forward(transferred_sentence_str, style_label)
        ###return rewards[:, -1]
        return rewards # batch size 크기의 1차원 리스트(?)

    @beartype
    def load(self, path: Optional[str] = None) -> None:
        """Load the model from the path

        Args:
            path (str): path to the model
        """
        if path is None:
            path = self.config.model_folder
            if os.path.exists(self.config.model_folder) is False:
                os.makedirs(self.config.model_folder)
                print(
                    f"Model folder does not exist. Creating it,"
                    f"and returning without loading the model:\n{path}"
                )
                return
        # load the model and the tokenizer
        if os.path.exists(path) is False:
            print(
                f"Impossible to load the model:\n{path}\n"
                f"The path doesn't exist."
            )
            return
        
        ### 기학습한 모델 (BERT)
        # 모델 및  토크나이저 임포트
        model_name = self.config.model # "klue/bert-base"
        if self.config.subtask_name == 'Saturi':
            model_saved_path = self.config.model_folder
        elif self.config.subtask_name == 'Formality':
            model_saved_path = self.config.model_folder

        if model_saved_path == None:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_saved_path, num_labels=2)
            print(self.config.subtask_name, model_saved_path)

        self.sc_tokenizer = AutoTokenizer.from_pretrained(model_name) # "klue/bert-base"



    @beartype
    def save(self, path: Optional[str] = None) -> None:
        """Save the model to the path

        Args:
            path (Optional[str], optional): Path to store the model.
                Defaults to None.
        """
        if path is None:
            path = self.config.model_folder
            if os.path.exists(self.config.model_folder) is False:
                os.makedirs(self.config.model_folder)
        self.model.save_pretrained(path)


# just to keep namings consistent
CriticModel = RewardModel


class RewardDataset(Dataset): ###Style Classifier용 커스텀 데이터셋
    """Dataset class for the reward model"""
    def __init__(self, src_data, trg_data, now_tokenizer, max_tokens_len, mode="train", rand=False):
        self.total_dataset = src_data + trg_data # list
        self.total_label = [0]*len(src_data) + [1]*len(trg_data) # list
        self.tokenizer = now_tokenizer
        self.mode = mode
        self.max_tokens_len = max_tokens_len

        if rand:
            random.seed(42)  # 랜덤 시드 고정
            random.shuffle(self.total_dataset)
            random.seed(42)  # 랜덤 시드를 같은 값으로 다시 고정하여 라벨도 셔플 순서와 일치하도록 함
            random.shuffle(self.total_label)

    def __len__(self):
        return len(self.total_dataset)

    def __getitem__(self, idx):
        sentence = self.total_dataset[idx]
        inputs = self.tokenizer(sentence, padding='max_length', max_length=self.max_tokens_len, truncation=True, return_tensors="pt")
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



class RewardTrainer: # classifier_train.py 참고
    """Reward class to train the reward model

    Args:
        config (ConfigModel): Config parameters for the model

    Attributes:
        model (RewardModel): Reward model
        config (ConfigModel): Config parameters for the model
        optimizer (torch.optim): Optimizer for the model
        loss (torch.nn): Loss function for the model

    Methods:
        train: Train the reward model
        generate_user_input: Generate the user input for the LLM to evaluate a
            couple, (user_input, completion) and assing a score
        distillate: Parse the dataset and assign scores using LLMs
    """

    def __init__(self, config: ConfigReward) -> None:
        self.model = RewardModel(config)
        self.config = config
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.lr
        )
        self.loss_function = torch.nn.CrossEntropyLoss() # torch.nn.MSELoss()
        if not os.path.exists("./models"):
            os.mkdir("./models")
        self.training_stats = TrainingStats()
        self.validation_flag = False
        if config.validation_dataset_path is not None:
            self.validation_flag = True

    def train(
        self,
    ) -> None:
        """Train the reward model"""
        print("Start Training the Reward Model")
        # get config parameters
        train_dataset_path = self.config.train_dataset_path
        #validation_dataset_path = self.config.validation_dataset_path
        batch_size = self.config.batch_size
        epochs = self.config.epochs
        device = self.config.device

        # create dataloaders
        train_0, train_1 = []
        with open(train_dataset_path + 'train.0', 'r', encoding='utf-8') as file:
            for line in file:
                train_0.append(line.strip())
        
        with open(train_dataset_path + 'train.1', 'r', encoding='utf-8') as file2:
            for line in file2:
                train_1.append(line.strip())
        

        if self.validation_flag:
            train_dataset = RewardDataset(train_0[:int(len(train_0)*0.8)], train_1[:int(len(train_1)*0.8)], self.config.sc_tokenizer, self.config.max_tokens, mode="train", rand=True)
            eval_dataset = RewardDataset(train_0[int(len(train_0)*0.8):], train_1[int(len(train_1)*0.8):], self.config.sc_tokenizer, self.config.max_tokens, mode="train", rand=True)
        else:
            train_dataset = RewardDataset(train_0, train_1, self.config.sc_tokenizer, self.config.max_tokens, mode="train", rand=True)

        iteration_per_print = self.config.iteration_per_print # (사용 안됨)

        # compute the number of iterations
        logging_steps = int(len(train_dataset) / batch_size) # = n_iter

        # Early Stopping 콜백 생성
        early_stopping = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)
        output_dir = 'rlhf/formality/sc_trainer_new_20epoch'

        # TrainingArguments 설정
        training_args = TrainingArguments(
                                        output_dir=output_dir,
                                        num_train_epochs=self.config.epochs,
                                        learning_rate=self.config.lr,
                                        per_device_train_batch_size=batch_size,
                                        per_device_eval_batch_size=batch_size,
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

        trainer = Trainer(model=self.model,
                        args=training_args,
                        compute_metrics=compute_metrics,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        tokenizer=self.sc_tokenizer)

        torch.cuda.empty_cache()
        
        trainer.train()
        
        self.model.save_pretrained('/data/hyunkyung_lee/style_transfer/rlhf/formality/outputs/formality_style_classifier_new_20epoch/')
