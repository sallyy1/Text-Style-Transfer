import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def sample_3d(probs, temperature=1):
    '''probs.shape = (batch, seq_len, dim)'''
    sample_idx = torch.zeros(probs.size(0), probs.size(1)).to(device)
    sample_probs = torch.zeros(probs.size(0), probs.size(1)).to(device)
    if temperature != 1:
        temp = torch.exp(torch.div(torch.log(probs + 1e-20), temperature))
    else:
        temp = probs
    for i, s in enumerate(temp):
        temp_idx = torch.multinomial(s, 1)  # shape = (seq_len, 1)
        temp_probs = s.gather(1, temp_idx)  # shape = (seq_len, 1)
        sample_idx[i] = temp_idx.squeeze(1)
        sample_probs[i] = temp_probs.squeeze(1)

    return sample_probs, sample_idx.long()
        
        
def cal_sc_reward(transferred_sentence_str: list, sc_model, sc_tokenizer, style_label: int) -> list:
    # 해당 배치 내 변환(생성)한 문장들에 대한 Style-based score 계산
    batch_target_reward = []
    
    for trans_sen in transferred_sentence_str:
        trans_inputs = sc_tokenizer(trans_sen, padding=True, return_tensors="pt") # return_tensors="pt"
        input_id = trans_inputs['input_ids'].to(device)
        attention_mask = trans_inputs['attention_mask'].to(device)

        output = sc_model(input_id, attention_mask=attention_mask)
        pred_logits = output.logits
        predictions = F.softmax(pred_logits, dim=1) # tensor([[0.7079, 0.2921]])
        ###predicted_labels = torch.argmax(predictions, dim=1)

        ###print(trans_sen)
        ###print(predictions)
        ###print(predicted_labels)
        ###print()
        
        if style_label==0: # 경상도(0) -> 표준어(1) NO // 표준어(1) -> 경상도(0) YES!!
            target_reward = (predictions[:, 1] - predictions[:, 0]) # <class 'torch.Tensor'>, torch.Size([1])
        elif style_label==1: # 표준어(1) -> 경상도(0) // 경상도(0) -> 표준어(1) YES!!
            target_reward = (predictions[:, 0] - predictions[:, 1]) # <class 'torch.Tensor'>, torch.Size([1])
            
        batch_target_reward.append(target_reward) # Scalar 값을 List에 추가
        ###print("STYLE target reward 값: ", type(target_reward)) # torch.Size([1])
        ###print(target_reward.size())

            
    return batch_target_reward # List


def cal_bl_reward(transferred_sentence_str: list, target_sentence_str: list, device) -> torch.Tensor:
    '''Caculate BLEU-based reward'''
    # 해당 배치 내 변환(생성)한 문장들에 대한 Style-based score 계산    
    smooth = SmoothingFunction()
    bleus = []
    for hyp, ref in zip(transferred_sentence_str, target_sentence_str): # 각 transferred_sentence_str[idx] - target_sentence_str[idx] 쌍 간의 BLEU Score 계산
        bleus.append(sentence_bleu([ref], hyp,
                                   smoothing_function=smooth.method1))
    bleus = torch.FloatTensor(bleus).to(device)
    ###print("BLEU target sampling/greedy 값: ", type(bleus)) # torch.Size([16])
    ###print(bleus.size())

    return bleus



def cal_reward_loss(sample_probs, reward, idxs=None): # 인풋 reward 값이 "클수록"
    sample_probs = sample_probs.contiguous()
    sample_logprobs = torch.log(sample_probs).to(device)
    reward = reward.unsqueeze(1).contiguous()
    
    ###print(sample_logprobs.size()) # torch.Size([128, 1])
    ###print(reward.size()) # torch.Size([16, 1])
    
    if idxs is not None: # idxs: tgt(타겟 문장) 텐서에서 [엔드 토큰] 이전 인덱스
        batch_size, max_len = sample_probs.size()
        mask = torch.zeros(batch_size, max_len).to(device)
        for i, l in enumerate(idxs):
            mask[i, :l] = 1
        mask = mask.float().contiguous()
        output = -sample_logprobs * reward * mask # 손실 Loss 아웃풋 값이 작아짐
        output = (output.sum(-1)/mask.sum(-1)).mean()
    else:
        output = -sample_logprobs * reward
        output = output.mean()

    return output

# transferred_sentence_str = '성격이 일단 제가 쫌 막 드센 건 아닌데'
# target_sentence_str = '성격이 일단 제가 조금 막 드센 건 아닌데'
# print(len(transferred_sentence_str))
# print(len(target_sentence_str))

# bleu_reward = cal_bl_reward(transferred_sentence_str, target_sentence_str)
# print(bleu_reward)
# print(bleu_reward.size())