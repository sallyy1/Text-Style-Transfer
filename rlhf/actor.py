import json
import os

import torch
from beartype import beartype
# from beartype.typing import Optional, Tuple # ModuleNotFoundError: No module named 'beartype.typing'
from typing import Optional, Tuple # python 3.10부터는 Optional이 파이썬 내장 타입으로 지원됨. 이전 버전은 대신 Union 사용해야 함

from einops import rearrange
from torch.utils.data import Dataset, DataLoader
from config import ConfigActor
from utils import TrainingStats

# from chatllama.llama_model import load_model
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast, DataCollatorForSeq2Seq
from transformers.modeling_outputs import Seq2SeqLMOutput

random_seed = 42
torch.manual_seed(random_seed) # DataLoader shuffle 시 랜덤시드 설정

args_subtask = 'Saturi' # 'Saturi' or 'Formality'

class ActorModel(torch.nn.Module):
    """Actor model that generates the augmented prompt from the initial
    user_input. The aim is to train this model to generate better prompts.

    Attributes:
        model: The model from LLaMA to be used
        tokenizer: The LLaMA tokenizer
        max_model_tokens (int): Maximum number of tokens that the model can
            handle
        config (ConfigActor): Configuration for the actor model

    Methods:
        load: Load the model from a path
        save: Save the model to a path
        forward: Compute the action logits for a given sequence.
        generate: Generate a sequence from a given prompt
    """

    def __init__(self, config: ConfigActor) -> None:
        super().__init__()
        # load the model
        self.max_model_tokens = config.max_tokens
        
        ### 기학습한 모델 (BART)
        # 모델 및  토크나이저 임포트
        model_name = config.model # 'hyunwoongko/kobart'
        if config.subtask_name == 'Saturi':
            model_saved_path = config.model_folder
        elif config.subtask_name == 'Formality':
            model_saved_path = config.model_folder


        if model_saved_path == None:
            self.model = BartForConditionalGeneration.from_pretrained(model_name)
        else:
            self.model = BartForConditionalGeneration.from_pretrained(model_saved_path)
            print(config.subtask_name, model_saved_path)
            
        self.model.config.output_hidden_states = True # 설정
        self.lm_head = self.model.lm_head # SC 및 BL loss 계산 시 사용 (dim=768 -> vocab_size=30000 변환하는 레이어)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name, max_length=self.max_model_tokens) # 초기화
        
        # save config
        self.config = config

    def parameters(self, **kwargs):
        """Return the parameters of the model

        Args:
            **kwargs:
        """
        return self.model.parameters()

    
    def forward( # Actor 모델 강화학습 시 (Unparalled) ??
        self, sequences: torch.Tensor, sequences_mask: torch.Tensor
    ) -> torch.Tensor:
        """Generate logits to have probability distribution over the vocabulary
            of the actions

        Args:
            sequences (torch.Tensor): Sequences of states and actions used to
                    compute token logits for the whole list of sequences
            attention_mask (torch.Tensor): Mask for the sequences attention

        Returns:
            logits (torch.Tensor): Logits for the actions taken
        """

        # return model_output.logits        
        BART_outputs = self.model(input_ids=sequences, attention_mask=sequences_mask) # Model Forward (labels 시퀀스 없음. unparalled dataset)
        if self.config.debug:
            print("ActorModel.forward")
            print("model_output_logits shape", BART_outputs.logits.shape)
            ###print("model_output logits", BART_outputs.logits)
        return BART_outputs # .logits


    @beartype
    @torch.no_grad()
    def generate(
        self, states: torch.Tensor, state_mask: torch.Tensor
    ) -> Tuple:
        """Generate actions and sequences=[states, actions] from state
            (i.e. input of the prompt generator model)

        Args:
            state (torch.Tensor): the input of the user
            state_mask (torch.Tensor): Mask for the state input (for padding)

        Returns:
            actions (torch.Tensor): Actions generated from the state
            sequences (torch.Tensor): Sequences generated from the
                state as [states, actions]
        """
        max_sequence = states.shape[1]
        max_tokens = self.config.max_tokens + max_sequence
        temperature = self.config.temperature
        # What if the states + completion are longer than the max context of
        # the model?
        
        # BART 모델 generate
        # sequences: 기존 transferred_sentence
        device = self.config.device ### 디버깅
        self.model.to(device)
        states, state_mask = states.to(device), state_mask.to(device)
        
        sequences = self.model.generate(
            input_ids=states,
            attention_mask=state_mask,
            max_length=max_tokens,
            temperature=temperature,
        )
        
        actions = sequences[:, states.shape[1] :]  # noqa E203
        if self.config.debug:
            print("ActorModel.generate")
            print("state", states) ### input_ids
            print("state shape", states.shape)
            print("sequence shape", sequences.shape) ### transferred_sentence: 생성한 텐서
            print("sequence", sequences)
            print("actions shape", actions.shape) ### actions = sequences[:, states.shape[1] :]
            print("actions", actions)
        return actions, sequences

    @beartype
    def load(self, path: Optional[str] = None) -> None:
        """Load the model from the path

        Args:
            path (str): Path to the model
        """
        # if path is None:
        #     path = self.config.model_folder + "/" + self.config.model + ".pt"
        #     if os.path.exists(self.config.model_folder) is False:
        #         os.mkdir(self.config.model_folder)
        #         print(
        #             f"Impossible to load the model: {path}"
        #             f"The path doesn't exist."
        #         )
        #         return
        # # load the model
        # if os.path.exists(path) is False:
        #     print(
        #         f"Impossible to load the model: {path}"
        #         f"The path doesn't exist."
        #     )
        #     return
        # model_dict = torch.load(path)
        # self.model.load_state_dict(model_dict["model"])
        
        if path is None:
            path = self.config.model_folder
            if os.path.exists(self.config.model_folder) is False:
                os.mkdir(self.config.model_folder)
                print(
                    f"Impossible to load the model: {path}"
                    f"The path doesn't exist."
                )
                return
        # load the model
        if os.path.exists(path) is False:
            print(
                f"Impossible to load the model: {path}"
                f"The path doesn't exist."
            )
            return
        self.model = BartForConditionalGeneration.from_pretrained(path)
        self.model.config.output_hidden_states = True
        self.lm_head = self.model.lm_head

    @beartype
    def save(self, path: Optional[str] = None) -> None:
        """Save the model to the path

        Args:
            path (Optional[str], optional): Path to store the model.
                Defaults to None.
        """
        # if path is None:
        #     path = self.config.model_folder + "/" + self.config.model + ".pt"
        #     if os.path.exists(self.config.model_folder) is False:
        #         os.mkdir(self.config.model_folder)
        # torch.save({"model": self.model.state_dict()}, path)

        if path is None:
            path = self.config.model_folder
            if os.path.exists(self.config.model_folder) is False:
                os.mkdir(self.config.model_folder)        
        self.model.save_pretrained(path)


class ActorDataset(Dataset): # BART(seq2seq)용 커스텀 데이터셋
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



class ActorTrainer: # Actor 모델 초기 훈련 시
    """Used to pre-train the actor model to generate better prompts.

    Args:
        config (ConfigActor): Configuration for the actor model

    Attributes:
        config (ConfigActor): Configuration for the actor model
        model (ActorModel): Actor model
        loss_function (torch.nn.CrossEntropyLoss): Loss function
        optimizer (torch.optim.Adam): Optimizer
        validation_flag (bool): Flag to indicate if the validation dataset
            is provided
        training_stats (TrainingStats): Training statistics

    Methods:
        train: Train the actor model
    """

    def __init__(self, config: ConfigActor) -> None:
        # load the model
        self.config = config
        self.model = ActorModel(config)
        self.loss_function = torch.nn.CrossEntropyLoss() # default loss function은 C.E Loss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.lr
        )
        self.validation_flag = False
        self.training_stats = TrainingStats()
        if not os.path.exists(config.model_folder):
            os.mkdir(config.model_folder)
        if config.validation_dataset_path is not None:
            self.validation_flag = True

    def train(
        self,
    ) -> None:
        print("Start Actor Model Pretraining")
        # get config parameters
        train_dataset_path = self.config.train_dataset_path
        validation_dataset_path = self.config.validation_dataset_path
        batch_size = self.config.batch_size
        epochs = self.config.epochs
        device = self.config.device

        # create dataloaders
        # 데이터 로드
        train_0 = []

        soure_path = train_dataset_path + '/train.0' # 소스 문장 (변환생성 전)

        with open(soure_path, 'r', encoding='utf-8') as file:
            for line in file:
                train_0.append(line.strip())


        train_1 = []

        target_path = train_dataset_path + '/train.1' # 타겟 문장 (변환생성 후)

        with open(target_path, 'r', encoding='utf-8') as file2:
            for line in file2:
                train_1.append(line.strip())
       
        # Get Data Collator
        args_fp16 = 8 # False
        seq2seq_datacollator = DataCollatorForSeq2Seq(self.model.tokenizer, self.model,
                                                        label_pad_token_id=self.model.tokenizer.pad_token_id,
                                                        pad_to_multiple_of=8 if args_fp16 else None)

        
        train_dataset = ActorDataset(train_0[:24000], train_1[:24000], self.model.tokenizer, mode="train")
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=seq2seq_datacollator)
        print("Train Dataloader: ", len(train_dataloader))
        if self.validation_flag:
            eval_dataset = ActorDataset(train_0[int(len(train_0)*0.8):int(len(train_0)*0.8)+6000], train_1[int(len(train_0)*0.8):int(len(train_0)*0.8)+6000], self.model.tokenizer, mode="train")
            validation_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=seq2seq_datacollator)
            print("Valid Dataloader: ", len(validation_dataloader))

        # compute the number of iterations
        n_iter = int(len(train_dataset) / batch_size)

        # traing loop
        for epoch in range(epochs):
            self.model.train()
            for i, input_output in enumerate(train_dataloader):
                print(len(input_output))
                training_input = input_output['input_ids'].to(device) ### 데이터(텐서) 이동시킴
                attention_mask = input_output['attention_mask'].to(device)
                trg_input_ids = input_output['decoder_input_ids'].to(device)
                trg_attention_mask = input_output['decoder_attention_mask'].to(device)

                labels = trg_input_ids.clone().detach() # Tensor 복사        
                # new_tensor = torch.full([args_batch_size, 1], 3).to(device)
                new_tensor = torch.full_like(labels[:, :1], 3).to(device)
                training_output = torch.cat((labels[..., 1:], new_tensor), dim=-1).contiguous()


                # forward pass
                BART_outputs = self.self.model(input_ids=training_input, attention_mask=attention_mask, decoder_input_ids=trg_input_ids, labels=training_output) # Teacher Forcing O
                loss = BART_outputs.loss
                self.training_stats.training_loss.append(loss.item())

                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # print progress
                if i % self.config.iteration_per_print == 0:
                    print(
                        f"Epoch: {epoch+1}/{epochs}, "
                        f"Iteration: {i+1}/{n_iter}, "
                        f"Training Loss: {loss}"
                    )
            if self.validation_flag:
                self.model.eval()
                for i, input_output in enumerate(validation_dataloader):
                    validation_input = input_output['input_ids'].to(device) ### 데이터(텐서) 이동시킴
                    attention_mask = input_output['attention_mask'].to(device)
                    trg_input_ids = input_output['decoder_input_ids'].to(device)
                    trg_attention_mask = input_output['decoder_attention_mask'].to(device)

                    # 추론 시
                    # (1) Cross Entropy loss
                    labels = trg_input_ids.clone().detach() # Tensor 복사        
                    # new_tensor = torch.full([args_batch_size, 1], 3).to(device)
                    new_tensor = torch.full_like(labels[:, :1], 3).to(device)
                    validation_output = torch.cat((labels[..., 1:], new_tensor), dim=-1).contiguous()
                    

                    # forward pass
                    BART_outputs = self.model(input_ids=validation_input, attention_mask=attention_mask, labels=validation_output) # Teacher Forcing X
                    loss = BART_outputs.loss                    
                    self.training_stats.validation_loss.append(loss.item())

                    # print progress
                    if i % self.config.iteration_per_print == 0:
                        print(
                            f"Epoch: {epoch+1}/{epochs}, "
                            f"Iteration: {i+1}/{n_iter}, "
                            f"Validation Loss: {loss}"
                        )
        self.model.save()
        print("Training Finished ")
