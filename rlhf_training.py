from rlhf.trainer import RLTrainer
from rlhf.config import Config

path = "rlhf/config.yaml" # "path_to_config_file.yaml"
config = Config(path=path)
trainer = RLTrainer(config)
# rlhf 폴더/trainer.py의 RLTrainer 클래스
# def __init__: ActorCritic(), RewardModel() 모델 선언, TrainingStats(), ConversationLog(), ExamplesSampler()
trainer.train()
# def train: RL 학습이 이뤄지는 함수. self.config.trainer.num_episodes만큼 for문 수행. 각 num_episodes에서는 self.config.trainer.max_timesteps만큰 for문 수행하며 PPO 알고리즘-> memories에 전체 episode 별 timestep별 states/actions/sequences/values/rewards/actions_log_probs/sequence_mask 저장됨 
trainer.training_stats.plot()