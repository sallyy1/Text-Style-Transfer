import sys
sys.path.append('/home/hyunkyung_lee/saturi_transfer/rlhf')


import json
import os
import random
from collections import deque, namedtuple

import torch
from beartype import beartype
# from beartype.typing import Deque, Tuple, List # ModuleNotFoundError: No module named 'beartype.typing'
from collections import deque
from typing import Tuple, List
from einops import rearrange
from torch.utils.data import Dataset, DataLoader

from actor import ActorModel
from reward import RewardModel, CriticModel
from config import ConfigReward, ConfigActor, Config
from utils import TrainingStats, ConversationLog

from custom_utils import sample_3d
import torch.nn.functional as F

class ActorCritic(torch.nn.Module): ###
    """Actor Critic class stores both the actor and the critic models
    and it generates values and action for given sequences during the training
    of the actor.

    Attributes:
        actor (ActorModel): Actor model
        critic (CriticModel): Critic model
        debug (bool): enable prints for Debugging

    Methods:
        forward: given a sequence returns action logits and values (used
            to evaluate the actor during training)
        generate: given a sequence returns action, action logits, values
            sequences and sequences masks (used to generate new sequences
            during acting phase)
    """

    def __init__(
        self, actor_config: ConfigActor, critic_config: ConfigReward
    ) -> None:
        super().__init__()
        self.actor = ActorModel(actor_config)
        self.critic = CriticModel(critic_config) # CriticModel = RewardModel
        self.debug = actor_config.debug

    @beartype
    def forward( # actorcritic 클래스 forward
        self,
        sequences: torch.Tensor, # Unparalled??
        sequences_mask: torch.Tensor,
        action_len: int,
    ) -> Tuple:
        """Given the whole sequences, use the actor forward to get the logits
            for each token in the sequence and the critic forward to get the
            values for each generation step.

        Args:
            sequences (torch.Tensor): Sequences composed of [states, actions]
            sequence_mask (torch.Tensor): Mask for the sequences
            action_length (int): Length of the actions in the sequences

        Returns:
            action_logits (torch.Tensor): Logits for the actions in the
                sequences
            values (torch.Tensor): Values for the actions in the sequences
        """
        # use a single forward on the whole sequence
        # to get pi(y | x) and ignore predicted output
        BART_outputs = self.actor.forward(sequences, sequences_mask) ### actor 모델 forward
        actions_logits = BART_outputs.logits
        
        # y^s 샘플링
        hidden_states = BART_outputs.decoder_hidden_states
        last_hidden_state = hidden_states[-1] # 최종 생성된 문장의 hidden states: (batch_size, sequence_length, config.d_model)
        #print("last_hidden_state 결과: ", last_hidden_state.size())
        
        ###out = F.softmax(last_hidden_state.float(), dim=-1) # torch.Size([16, 20]) = (batch_size, seq_len) ### 확인 필요 !!!
        
        # dim=768 -> vocab idx 변환
        vocab_idx_outputs = self.actor.lm_head(last_hidden_state)
        out = F.softmax(vocab_idx_outputs.float(), dim=-1)
        sample_probs, sample_idx = sample_3d(out)     

        # -> 디코딩된 문장 변환
        sampling_decoded_text = self.actor.tokenizer.batch_decode(sample_idx, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        ###print(sampling_decoded_text) # [16]

        
        sampling_values = self.critic.forward(sampling_decoded_text, style_label=0) ### critic 모델 forward
        # 경상도(0) -> 표준어(1) 변환 시 # 합쇼체(0) -> 반말체(1) 변환 시
        # sampling한 생성 결과를 Style Classifier로 추론했을 때 R 값

        # return only logits and values for the actions taken
        ### 0915 디버깅
        print("actions_logits 크기:", actions_logits.size()) ### torch.Size([2, 18, 30000])
        print("values 크기:", sampling_values.size()) ### torch.Size([2])
        real_actions_logits = actions_logits[:, -action_len:, :]
        ###real_values = values[:, -action_len:]
        real_values = sampling_values[:]
        print("real_actions_logits 크기:", real_actions_logits.size())
        print("real_values 크기:", real_values.size())

        if self.debug:
            print("ActorCritic.forward")
            print("action_len", action_len)
            print("sequences.shape", sequences.shape)
            print("sequences", sequences)
            print("real_action_logits.shape", actions_logits.shape)
            print("real_action_logits", actions_logits)
            print("real_values.shape", sampling_values.shape)
            print("real_values", sampling_values)

        return (
            real_actions_logits,
            real_values,
            sample_probs,
            sampling_decoded_text,
        )

    @torch.no_grad()
    @beartype
    def generate(
        self, states: torch.Tensor, state_mask: torch.Tensor
    ) -> Tuple:
        """Generate actions, actions_logits, values and sequences from states

        Args:
            states (torch.Tensor): user inputs
            state_mask (torch.Tensor): Mask for the states of the environment

        Returns:
            actions (torch.Tensor): Actions generated from the states
            actions_logits (torch.Tensor): Logits for the actions generated
                from the states (i.e. pi(y | x))
            values (torch.Tensor): Values generated by the critic model
                for the actions generated by the actor (i.e. V(x))
            sequences (torch.Tensor): Sequences generated from the states
                as [states, actions]
        """
        ### (args_infer_fashion == 'batch' 배치 기준일 때)
        # generate action sequence
        device = 'cuda'
        states, state_mask = states.to(device), state_mask.to(device)
        actions, sequence = self.actor.generate(states, state_mask) ### generate
        sequences_mask = sequence != self.actor.tokenizer.pad_token_id
        action_len = actions.shape[1]

        # generate actions_logits and values
        actions_logits, values, sample_probs, sampling_decoded_text = self.forward( ### forward
            sequence, sequences_mask, action_len ### transferred_sentence: 생성한 텐서
        )
        if self.debug:
            print("ActorCritic.generate")
            print("actions shape", actions.shape)
            print("actions", actions)
            print("sequence shape", sequence.shape)
            print("sequence", sequence)
            print("actions_logits shape", actions_logits.shape)
            print("actions_logits", actions_logits)
            print("values shape", values.shape)
            print("values", values)

        return actions, actions_logits, values, sequence, sequences_mask


# structure to store the data for each experience
Memory = namedtuple(
    "Memory",
    [
        "states",
        "actions",
        "sequences",
        "values",
        "rewards",
        "actions_log_probs",
        "sequences_mask",
    ],
)


class ExperienceDataset(Dataset): # Unparalled data
    """Dataset to train the actor-critic models"""

    def __init__(
        self,
        # memories: Deque[Memory],
        memories: List[Memory],
        device: torch.device,
    ) -> None:
        super().__init__()
        self.data = list(memories)
        self.device = device

    def __len__(
        self,
    ) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple:
        # return the idx-th memory element as a tuple of tensors on the device
        item = (
            self.data[idx].states.to(self.device),
            self.data[idx].actions.to(self.device),
            self.data[idx].sequences.to(self.device),
            self.data[idx].values.to(self.device),
            self.data[idx].rewards.to(self.device),
            self.data[idx].actions_log_probs.to(self.device),
            self.data[idx].sequences_mask.to(self.device),
        )
        return item


class ExamplesSampler:
    """Store the prompt to be sampled to generate the examples
    read a json file with the following format:
    [
        {
            "user_input" : "",
        } ,
        ...
    ]
    Where:
        user_input: is the input of the user or directly the input of the user
            with the memory preappended (i.e. user_input + memory)
    """

    def __init__(
        self,
        path: str,
    ) -> None:
        self.path = path
        self.data = []
        with open(path, "r", encoding='utf-8') as f: # unparalled.0 파일 열기
            # self.data = json.load(f)
            for line in f:
                self.data.append(line.strip())

    def sample(self, n: int) -> List:
        """Sample n examples from the data

        Args:
            n (int): Number of examples to sample
        """
        return random.sample(self.data, n)


class RLTrainer: ###
    """Train the actor-critic model using RL

    Attributes:
        config (Config): Configuration of the trainer
        debug (bool): Debug mode
        actorcritic (ActorCritic): Actor-critic model
        actor_optim (torch.optim): Optimizer for the actor
        critic_optim (torch.optim): Optimizer for the critic
        reward (RewardModel): Reward model
        training_stats (TrainingStats): Class to store training stats
    Methods:
        train: the training loop that calls the learn function after generating
            the experiences.
        learn: Learn from a batch of experiences and update the actor and the
            critic model.
        load_checkpoint: Load the checkpoint of the actor-critic model
        save_checkpoint: Save the checkpoint of the actor-critic model
        generate_user_input: Generate the user input from the inputs
    """

    def __init__(
        self,
        config: Config, # config.py의 ConfigTrainer 클래스, ConfigActor 클래스, ConfigReward 클래스, Config 클래스
    ) -> None:
        
        # self.config = config
        self.debug = config.trainer.debug
        
        self.config = config.trainer # config.py의 ConfigTrainer 클래스
        ###self.config_actor_critic = config # config.py의 ConfigTrainer 클래스, ConfigActor 클래스, ConfigReward 클래스, Config 클래스
        # self.debug = config.debug

        # initialize agent-critic
        print(config)
        self.actorcritic = ActorCritic(config.actor, config.critic) # ActorCritic 클래스 선언
        #self.actorcritic = ActorCritic(config.config_actor_critic, config.config_actor_critic)
        self.actor_optim = torch.optim.Adam( # actor 모델
            self.actorcritic.actor.parameters(), lr=config.trainer.actor_lr
        )
        self.critic_optim = torch.optim.Adam( # critic 모델
            self.actorcritic.critic.parameters(), lr=config.trainer.critic_lr
        )

        # initialize reward model
        self.reward = RewardModel(config.reward) # Critic 모델 선언

        # initialize class to store training stats
        self.training_stats = TrainingStats()
        self.conversation_log = ConversationLog()

        # initialize examples sampler
        self.example_sampler = ExamplesSampler(config.trainer.examples_path)

        # eps
        self.eps = 1e-3 ###1e-8

        # make models directory
        if not os.path.exists("./models"):
            os.mkdir("./models")

        # if not os.path.exists(self.config.trainer.checkpoint_folder):
        #     os.mkdir(self.config.trainer.checkpoint_folder)
        if not os.path.exists(self.config.checkpoint_folder):
            os.mkdir(self.config.checkpoint_folder)

    def save_checkpoint(
        self,
        current_episode: int,
    ) -> None:
        print(f"Saving checkpoint for episode {current_episode+1}..")
        file_name = "rltraining_" + str(current_episode) + ".pt"
        checkpoint_folder = self.config.trainer.checkpoint_folder
        if os.path.exists(checkpoint_folder) is False:
            os.mkdir(checkpoint_folder)
        path = checkpoint_folder + "/" + file_name
        torch.save(
            {
                "episode": current_episode,
                "actor_state_dict": self.actorcritic.actor.state_dict(), # actor 모델
                "critic_state_dict": self.actorcritic.critic.state_dict(), # critic 모델
                "actor_optim_state_dict": self.actor_optim.state_dict(), # actor 모델
                "critic_optim_state_dict": self.critic_optim.state_dict(), # critic 모델
                "training_stats": self.training_stats,
            },
            path,
        )

    def load_checkpoint(
        self,
    ) -> int:
        # get all the files name in the checkpoint folder and take the one
        # with the highest epoch
        checkpoint_folder = self.config.checkpoint_folder
        if os.path.exists(checkpoint_folder) is False:
            os.mkdir(checkpoint_folder)
            print(
                f"Checkpoint folder {checkpoint_folder} does not exist.\n"
                f"No checkpoint will be loaded."
            )
            return
        files = os.listdir(checkpoint_folder)
        episodes = [int(f.split("_")[1].split(".")[0]) for f in files]
        if len(episodes) == 0:
            return 0
        max_episode = max(episodes)
        print(f"Loading checkpoint for episode {max_episode+1}..")
        file_name = "rltraining_" + str(max_episode) + ".pt"
        path = checkpoint_folder + "/" + file_name
        checkpoint = torch.load(path)
        self.actorcritic.actor.load_state_dict(checkpoint["actor_state_dict"]) # actor 모델
        self.actorcritic.critic.load_state_dict( # critic 모델
            checkpoint["critic_state_dict"]
        )
        self.actor_optim.load_state_dict(checkpoint["actor_optim_state_dict"])
        self.critic_optim.load_state_dict(
            checkpoint["critic_optim_state_dict"]
        )
        self.trainign_stats = checkpoint["training_stats"]
        self.actorcritic.actor.to(self.config.trainer.device) # actor 모델
        self.actorcritic.critic.to(self.config.trainer.device) # critic 모델
        return max_episode + 1  # return the next episode to train

    @beartype
    # def learn(self, memories: Deque[Memory]) -> None:
    def learn(self, memories: List[Memory]) -> None:
        # 여기서 이번 episode에서 수행한 time step만큼의 결과값(LM 모델의 action & RM 모델의 평가값) 가지고 PPO 계산해서
        # self.actorcritic 모델 업데이트 됨
        # -> self.actorcritic 모델: class ActorCritic(torch.nn.Module)에서 actor와 critic 모델을 입력받아 self.actorcritic.forward() 함수에서는 action_logits와 values를 리턴함
        """Train the agent-critic model using RL:
        - for each batch of episodes, compute action logits and values
        - then compare action logits probs with memories one and values with
            rewards to compute the PPO loss and update the actor-critic model
        """
        print("Start to Learn...")

        epochs = self.config.epochs
        actor_eps_clip = self.config.actor_eps_clip
        critic_eps_clip = self.config.critic_eps_clip
        beta_s = self.config.beta_s
        batch_size = self.config.batch_size
        device = self.config.device
        

        # create dataset from memories
        dataloader = DataLoader(
            ExperienceDataset(memories, device), batch_size=batch_size
        )

        # train agent-critic
        self.actorcritic.train() # 학습 모드
        for epoch in range(epochs):
            for i, (
                states,
                old_actions,
                sequences,
                old_values,
                rewards,
                old_actions_log_probs,
                sequences_mask,
            ) in enumerate(dataloader):

                # print
                print(
                    "Epoch",
                    epoch + 1,
                    "of",
                    epochs,
                    "Data",
                    i + 1,
                    "of",
                    int(len(dataloader) / batch_size),
                )

                if self.debug:
                    print("RLTrainer.learn()")
                    print("memory states shapes are: ")
                    print("states shape", states.shape)
                    print("old_actions shape", old_actions.shape)
                    print("sequences shape", sequences.shape)
                    print("old_values shape", old_values.shape)
                    print("rewards shape", rewards.shape)
                    print(
                        "old_actions_log_probs shape",
                        old_actions_log_probs.shape,
                    )
                # reshaping rewards to match [b, s] shape
                rewards = rearrange(rewards, "b -> b 1")

                # get actions len
                actions_len = old_actions.shape[-1]

                # get actor critic forward
                actions_logits, sampling_values, sample_probs_batch, sampling_decoded_text_batch = self.actorcritic.forward( ### ActorCritic 클래스의 forward 함수: 여기서 actor 모델과 critic 모델 포워드
                    sequences, sequences_mask, actions_len ### sequences: input_ids(?), sequences_mask: attention_mask(?)
                ) # actor 모델에서 sampling 텐서는 1개만 뽑았었음 (actor 모델의 input 배치=1 ???)

                ##########################
                ### [1] actor 모델 loss 계산  
                ### 틀린 코드 (N개 샘플링하는 거 아님)
                '''
                # # 샘플링할 데이터 수 (N)
                # N = batch_size
                # #####N = batch_size // 2 ### N 몇으로 설정?

                # # batch_size에서 N개의 인덱스 추출
                # random_indices = random.sample(range(batch_size), N)

                # # 각 텐서에서 추출된 인덱스에 해당하는 데이터 추출
                # N_actions_logits = actions_logits[random_indices].to(device)  # (N, num_actions)
                # N_sampling_values = sampling_values[random_indices].to(device)  # (N,)
                # N_sample_probs_batch = sample_probs_batch[random_indices].to(device)  # (N, num_samples)
                # N_sampling_decoded_text_batch = [sampling_decoded_text_batch[idx] for idx in random_indices]  # (N, num_samples)

                # ###if self.debug:
                # # 각 텐서의 크기 확인
                # print("Sampled Actions Logits Size:", N_actions_logits.size()) # torch.Size([1, 66, 30000])
                # print("Sampled Values Size:", N_sampling_values.size()) # torch.Size([1])
                # print("Sampled Sample Probs Batch Size:", N_sample_probs_batch.size()) # torch.Size([1, 66])
                # print("Sampled Sampling Decoded Text Batch Size:", len(N_sampling_decoded_text_batch)) # 1
                # print(N_sampling_decoded_text_batch)
                
                # # sampled_values와 log를 씌운 sampled_actions_logits를 곱하여 기댓값 계산
                # loss = (N_sampling_values * N_sample_probs_batch.exp()).mean()
                '''


                ###if self.debug:
                # 각 텐서의 크기 확인
                #####actions_logits, sampling_values, sample_probs_batch, sampling_decoded_text_batch = actions_logits.to(device), sampling_values.to(device), sample_probs_batch.to(device), sampling_decoded_text_batch.to(device)
                actions_logits, sampling_values, sample_probs_batch = actions_logits.to(device), sampling_values.to(device), sample_probs_batch.to(device)


                # transferred_sentence_str = [self.actorcritic.actor.tokenizer.decode(sentence, skip_special_tokens=True) for sentence in actions]
                
                # input_sentence_str = [self.actorcritic.actor.tokenizer.decode(sentence, skip_special_tokens=True) for sentence in states]

                # print("인풋한 텐서: ", states[0]) ### input_ids
                # print("디코더 인풋 텐서: ", trg_input_ids[0])
                # print("생성한 텐서: ", actions[0]) ### transferred_sentence
                # (sequences가 배치 단위일 때)
                # transferred_sentence_str = [self.actorcritic.actor.tokenizer.decode(sentence, skip_special_tokens=True) for sentence in actions]
                
                # input_sentence_str = [self.actorcritic.actor.tokenizer.decode(sentence, skip_special_tokens=True) for sentence in states]
                # print("- * - * - 인풋한 문장(배치): \n" + "\n".join(s for s in input_sentence_str) + "\n")
                # print("- * - * - 변환한 문장(배치): \n" + "\n".join(s for s in transferred_sentence_str))
          
                # (sequences가 한 문장 단위일 때)
                print(sequences)
                input_sentence_str = self.actorcritic.actor.tokenizer.decode(sequences[0].tolist(), skip_special_tokens=True)
                # print("- * - * - 인풋한 문장(각 문장): \n" + input_sentence_str + "\n")          
                # transferred_sentence_str = self.actorcritic.actor.tokenizer.batch_decode(transferred_sentence, skip_special_tokens=True)[0]
                # print("- * - * - 인풋한 문장(각 문장): \n" + input_sentence_str + "\n")
                # print("- * - * - 변환한 문장(각 문장): \n" + transferred_sentence_str + "\n")
       

                print("인풋 문장(경상도): ", input_sentence_str)
                print("Sampled Actions Logits Size:", actions_logits.size()) # torch.Size([1, 66, 30000])
                print("Sampled Values Size:", sampling_values.size()) # torch.Size([1])
                print("Sampled Sample Probs Batch Size:", sample_probs_batch.size()) # torch.Size([1, 66])
                print("Sampled Sampling Decoded Text Batch Size:", len(sampling_decoded_text_batch)) # 1
                print("변환 문장(표준어): ", sampling_decoded_text_batch)
                
                # sampled_values와 log를 씌운 sampled_actions_logits를 곱하여 기댓값 계산
                loss = (sampling_values * sample_probs_batch.exp()).mean()                
                

                # get action log prob
                # actions_prob = (
                #     torch.softmax(actions_logits, dim=-1).max(dim=-1).values
                # )
                # actions_log_prob = torch.log(actions_prob + self.eps)
                
                # check if loss item is nan
                if torch.isnan(loss):
                    raise ValueError("Loss is nan")
                print("loss", loss.item()) # loss -1.3637319803237915

                if self.debug:
                    print("values", values)
                    print("old_values", old_values)
                    print("rewards", rewards)
                    print("ratios", ratios)
                    print("advantages", advantages)
                    print("entropies", entropies)
                '''
                ### [1] actor 모델 loss 계산

                # get action log prob
                actions_prob = (
                    torch.softmax(actions_logits, dim=-1).max(dim=-1).values
                )
                actions_log_prob = torch.log(actions_prob + self.eps)

                # compute entropy
                entropies = (actions_prob * actions_log_prob).sum(dim=-1)

                # compute KL divergence
                kl_div_loss = (
                    (actions_prob * (old_actions_log_probs - actions_log_prob))
                    .sum(dim=-1)
                    .mean()
                )

                # compute PPO Loss -- Whan dimensions are different
                # (especially the values and the probs are
                #  multiplied directly with the reward)
                ratios = (actions_log_prob - old_actions_log_probs).exp()
                advantages = rewards - old_values
                # normalize advantages
                advantages = (advantages - advantages.mean(dim=-1)) / (
                    advantages.std() + self.eps
                )
                surr1 = advantages * ratios
                surr2 = (
                    torch.clamp(ratios, 1 - actor_eps_clip, 1 + actor_eps_clip)
                    * advantages
                )
                policy_loss = -torch.min(surr1, surr2) - beta_s * entropies
                policy_loss = policy_loss.mean()
                loss = policy_loss + kl_div_loss
                print(" * * * * * * * * * * * * * * * ")
                print("actions_prob :", actions_prob)
                print("actions_log_prob :", actions_log_prob)
                print("")
                print("entropies :", entropies)
                print("policy_loss: ", policy_loss)
                print("kl_div_loss :", kl_div_loss)
                # check if loss item is nan
                if torch.isnan(loss):
                    raise ValueError("Loss is nan")
                print("loss", loss.item())

                if self.debug:
                    print("values", values)
                    print("old_values", old_values)
                    print("rewards", rewards)
                    print("ratios", ratios)
                    print("advantages", advantages)
                    print("entropies", entropies)
                '''

                # update actor with loss
                self.actor_optim.zero_grad() ### [1] actor 모델 optimizer 업데이트
                loss.backward()
                self.actor_optim.step()

                torch.cuda.synchronize(device)

                ###########################
                ### [2] critic 모델 loss 계산
                # compute value loss
                '''
                values = sampling_values[:]
                value_loss_clipped = old_values + (values - old_values).clamp(
                    -critic_eps_clip, critic_eps_clip
                )
                value_loss1 = (value_loss_clipped - rewards) ** 2
                value_loss2 = (values - rewards) ** 2
                value_loss = torch.max(value_loss1, value_loss2).mean()
                if torch.isnan(value_loss):
                    raise ValueError("Value loss is nan")
                print("value_loss", value_loss.item()) # 0.28172463178634644

                # upate critic with loss
                self.critic_optim.zero_grad() ### [2] critic 모델 optimizer 업데이트
                value_loss.backward() ### value_loss는 값이여서 백워드 할 수 없음
                self.critic_optim.step()

                self.training_stats.training_loss.append(
                    loss.detach().cpu().item()
                )
                self.training_stats.value_loss.append(
                    value_loss.detach().cpu().item()
                )
                '''

        self.actorcritic.eval() ### ActorCritic 클래스에 eval 함수는 없는데? 기본 내장 함수 사용?
        print("End Learning")

    def train(
        self,
    ) -> None:
        # initialize settings
        num_episodes = self.config.num_episodes
        max_timesteps = self.config.max_timesteps
        num_examples = self.config.num_examples
        update_timesteps = self.config.update_timesteps
        batch_size = self.config.batch_size
        update_checkpoint = self.config.update_checkpoint
        device = self.config.device

        print("Start RL Training")
        # check dimensions consistency
        # at each time step num_examples memories are generated
        number_of_memories_per_learn_iteration = (
            num_examples * update_timesteps
        )
        # the number of memories must be a multiple of the batch size
        assert (
            number_of_memories_per_learn_iteration % batch_size == 0
        ), "The number of memories must be a multiple of the batch size"
        # the total number of timesteps is
        total_number_of_timesteps = num_episodes * max_timesteps
        # the update_timesteps must be a multiple
        #  of the total number of timesteps
        assert total_number_of_timesteps % update_timesteps == 0, (
            "The number of timesteps (num_episodes*max_timesteps)"
            "must be a multiple of the update_timesteps"
        )

        # initialize memories
        memories = deque([])

        # loop over episodes and timesteps
        current_time = 0
        checkpoint_counter = 0
        current_episode = self.load_checkpoint()
        current_learn_counter = 0

        self.actorcritic.eval() # 추론 모드  ### ActorCritic 클래스에 eval 함수는 없는데? 기본 내장 함수 사용?
        for eps in range(current_episode, num_episodes):
            for timestep in range(max_timesteps):

                print(
                    f"Episode: {eps + 1} of {num_episodes}, "
                    f"Timestep: {timestep + 1} of {max_timesteps}",
                )

                # counter used to count timesteps into memory
                current_time += 1

                # sample num_examples examples from  example dataset
                inputs = self.example_sampler.sample(num_examples) ### 샘플은 왜 하는지? 꼭 해야 하는지?

                # tokenize examples
                tokenized_inputs = self.actorcritic.actor.tokenizer(
                    inputs, padding=True, return_tensors="pt"
                )
                if self.debug:
                    print("RLTrainer.train()")
                    print("tokenized inputs", tokenized_inputs)
                # states are [batch_size, seq_len_of_states]
                # 시퀀스 input_ids & attention_mask (배치인지는 sampler 확인해봐야 함)
                states = tokenized_inputs["input_ids"].to(device)
                states_mask = tokenized_inputs["attention_mask"].to(device)

                # generate prompts
                # actions --> output produced by the actor head in response
                #  of the state(input) [batch_size, len_of_actions]
                # actions_logits --> logits of the actions
                # [batch_size, len_of_actions, vocab_size]
                # values --> output produced by the critic for each action
                # [batch_size, len_of_actions]
                # sequence --> (state, actions)
                # [batch_size, len_of_actions + seq_len_of_states] =
                # [batch_size, seq_len]
                (
                    actions,
                    actions_logits,
                    values,
                    sequences,
                    sequences_mask,
                ) = self.actorcritic.generate(states, states_mask)  ### ActorCritic 클래스의 generate 함수

                # from action logits to action log probs
                action_prob = (
                    torch.softmax(actions_logits, dim=-1).max(dim=-1).values
                )
                actions_log_probs = torch.log(action_prob + self.eps)

                ### [1] actor 모델 추론 결과인 action을 디코딩된 문장으로 변환
                ### 생성 모델 안쓸 거임
                # completions = [
                #     self.actorcritic.actor.tokenizer.decode(action)
                #     for i, action in enumerate(actions) ### actions는 배치단위 맞을 듯
                # ]
                
                
                print("인풋한 텐서: ", states[0]) ### input_ids
                # print("디코더 인풋 텐서: ", trg_input_ids[0])
                print("생성한 텐서: ", actions[0]) ### transferred_sentence
                # (actions가 배치 단위일 때)
                transferred_sentence_str = [self.actorcritic.actor.tokenizer.decode(sentence, skip_special_tokens=True) for sentence in actions]
                
                input_sentence_str = [self.actorcritic.actor.tokenizer.decode(sentence, skip_special_tokens=True) for sentence in states]
                print("- * - * - 인풋한 문장(배치): \n" + "\n".join(s for s in input_sentence_str) + "\n")
                print("- * - * - 변환한 문장(배치): \n" + "\n".join(s for s in transferred_sentence_str))
          
                # (actions가 한 문장 단위일 때)
                '''
                input_sentence_str = self.actorcritic.actor.tokenizer.decode(input_ids[idx], skip_special_tokens=True)
                print("- * - * - 인풋한 문장(각 문장): \n" + input_sentence_str + "\n")          
                transferred_sentence_str = self.actorcritic.actor.tokenizer.batch_decode(transferred_sentence, skip_special_tokens=True)[0]
                # print("- * - * - 인풋한 문장(각 문장): \n" + input_sentence_str + "\n")
                print("- * - * - 변환한 문장(각 문장): \n" + transferred_sentence_str + "\n")
                '''
                




                ### [2] 디코딩된 문장 인풋하여 critic 모델 추론 -> reward 값 얻기
                ### 생성 모델 안 쓸거임
                # compute reward for the completion
                # the reward must take into account the answer quality wrt to
                # the initial request given
                # and must be tokenized again
                '''
                ### 생성 모델 용
                task_responses = []
                for input, completion in zip(inputs, completions):
                    task_response = input + "\n" + completion
                    task_responses.append(task_response)
                    
                    
                if self.debug:
                    print("RLTrainer.train()")
                    print("task_responses:")
                    for i, task_response in enumerate(task_responses):
                        print(i, task_response)
                        print("")
                tokenized_responses = self.reward.tokenizer(
                    task_responses, padding=True, return_tensors="pt"
                )
                # self.Reward(critic 모델) 통해 reward 값 추론
                rewards = self.reward.get_reward(
                    tokenized_responses["input_ids"].to(device),
                    tokenized_responses["attention_mask"].to(device),
                )
                '''
                
                # input_sentence_str는 이미 배치 단위
                    
                    
                if self.debug:
                    print("RLTrainer.train()")
                    print("input_sentence_str:")
                    for i, task_response in enumerate(input_sentence_str):
                        print(i, input_sentence_str)
                        print("")
                        
                        
                # self.Reward(critic 모델) 통해 reward 값 추론
                rewards = self.reward.get_reward(
                    input_sentence_str, # BART가 생성한 결과 문장 (디코딩된 문장)
                    style_label=0 # BERT Classifier에서 계산할 변환 태스크 번호 (0(경상도) -> 1(표준어))
                ) # rewards: 배치 단위



                ### 
                # store memories of the episode / timestep
                for i in range(states.shape[0]):
                    memories.append(
                        Memory(
                            *map(
                                lambda x: x.detach().cpu(),
                                (
                                    states[i, :],
                                    actions[i, :],
                                    sequences[i, :],
                                    values[i], ###values[i, :],
                                    rewards[i],
                                    actions_log_probs[i, :],
                                    sequences_mask[i, :],
                                ),
                            )
                        )
                    )

                print("= = = = = rewards 타입: ", type(rewards), type(rewards[i]), rewards[i])
                # log the memories in the conversation log
                for i in range(states.shape[0]):
                    ### (디버깅)
                    reward_value = rewards[i].item()  # Tensor를 float로 변환
                    # if i==0:
                    #     previous_reward = 0  # 초기 값으로 0을 사용
                    #     previous_completion = ""  # 초기 값으로 빈 문자열 사용
                    # else:
                    #     previous_reward = 0  # 초기 값으로 0을 사용
                    #     previous_completion = ""  # 초기 값으로 빈 문자열 사용
                    self.conversation_log.add_conversation(
                        inputs[i],
                        input_sentence_str[i], # completions[i],
                        reward_value, ###rewards[i].detach().cpu(),
                        current_learn_counter,
                    )

                # learn from memories
                print(
                    f"Learning counter: {current_time} of {update_timesteps}"
                )
                if (current_time % update_timesteps == 0) and (
                    current_time != 0
                ):
                    checkpoint_counter += 1
                    self.conversation_log.show(current_learn_counter)
                    memories = list(memories) ### deque 객체 -> list로 변환
                    self.learn(memories) # i번째 episode에서의 모든 timesteps에서 수행된 actor/critic 추론 결과 값 반영하여 학습에 적용
                    memories.clear() # i+1번째 episode로 넘어갈 때에는 memories 초기화
                    current_time = 0
                    current_learn_counter += 1

                if (checkpoint_counter % update_checkpoint == 0) and (
                    checkpoint_counter != 0
                ):
                    self.save_checkpoint(eps)
                    checkpoint_counter = 0

        # self.config.trainer.num_episodes 만큼 강화학습 수행 끝
        self.actorcritic.critic.save() # 모델 저장
        self.actorcritic.actor.save() # 모델 저장
        # print("Show conversations log")
        # self.conversation_log.show()
        print("End RL Training")
                                         