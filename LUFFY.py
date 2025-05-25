import numpy as np
import random
import torch
import torch.nn.functional as F
import copy
import math # 引入math模块

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# LUFFY 配置常量
GAMMA_POLICY_SHAPING = 0.1 # 策略塑造函数中的 gamma 参数
NUM_ON_POLICY_GENERATIONS = 7 # 每个提示的在线策略生成数量
NUM_OFF_POLICY_GENERATIONS = 1 # 每个提示的离线策略生成数量

def set_random_seed(seed: int = 42):
    """
    为Python、NumPy和PyTorch设置随机种子以确保可复现性。

    参数:
        seed (int): 要使用的种子值。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(42)

SYSTEM_PROMPT = """
Respond in the following format:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def prepare_dataset(split="train"):
    """加载并准备GSM8K数据集，用于字符串提示的训练。"""
    data = load_dataset('openai/gsm8k', 'main', trust_remote_code=True)[split]
    formatted_data = []
    for example in data:
        prompt_str = build_prompt([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]}
        ])
        formatted_example = {
            "prompt": prompt_str,
            "answer": extract_answer_from_dataset(example["answer"]),
            "raw_question": example["question"] # 为离策略模型保留原始问题
        }
        formatted_data.append(formatted_example)
    return formatted_data

def build_prompt(messages):
    """从消息列表构建单个提示字符串。"""
    return "\n".join([msg["content"].strip() for msg in messages])

def extract_answer_from_model_output(text):
    """从文本中最后一个<answer>标签提取值。"""
    parts = text.split("<answer>")
    if len(parts) < 2: return None
    last_part = parts[-1]
    if "</answer>" not in last_part: return None
    answer = last_part.split("</answer>")[0].strip()
    return None if answer == "..." else answer

def extract_answer_from_dataset(text):
    """从数据集中提取答案。"""
    if "####" not in text: return None
    return text.split("####")[1].strip()

def _extract_last_number(text):
    """从文本中提取最后一个数字。"""
    import re
    text = str(text).replace('$', '').replace('%', '').replace(',', '')
    pattern = r'-?\d+\.?\d*'
    numbers = re.findall(pattern, text)
    return float(numbers[-1]) if numbers else None

def _extract_single_number(text):
    """如果文本中只有一个数字，则提取它。"""
    import re
    text = str(text).replace('$', '').replace('%', '').replace(',', '')
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return float(numbers[0]) if len(numbers) == 1 else None

def evaluate_model(model, tokenizer, eval_examples, device, model_name="Model"):
    """在给定的一组样本上评估模型，并打印详细结果。"""
    model.eval()
    correct = 0
    total = len(eval_examples)
    print("\n" + "="*50)
    print(f"EVALUATION ON {total} EXAMPLES ({model_name})")
    print("="*50)
    
    for example in eval_examples:
        full_prompt = example["prompt"]
        expected = example["answer"]
        
        inputs = tokenizer.encode(full_prompt, return_tensors="pt").to(device) # Shape: (1, prompt_seq_len)
        outputs = model.generate(
            inputs,
            max_new_tokens=512,
            temperature=0.0, # 评估时使用确定性生成
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )
        response_ids = outputs[0] # Shape: (total_seq_len,)
        response = tokenizer.decode(response_ids[inputs.shape[1]:], skip_special_tokens=True)

        predicted = extract_answer_from_model_output(SYSTEM_PROMPT + "\n" + example['raw_question'] + "\n" + response) 
        
        is_correct = False
        if predicted is not None and expected is not None:
            if str(predicted).strip() == str(expected).strip():
                is_correct = True
            else:
                pred_num_last = _extract_last_number(predicted)
                exp_num_last = _extract_last_number(expected)
                if pred_num_last is not None and exp_num_last is not None and math.isclose(pred_num_last, exp_num_last):
                    is_correct = True
                else:
                    pred_num_single = _extract_single_number(predicted)
                    exp_num_single = _extract_single_number(expected)
                    if pred_num_single is not None and exp_num_single is not None and math.isclose(pred_num_single, exp_num_single):
                        is_correct = True
        
        if is_correct:
            correct += 1
            
        print(f"\nPrompt:\n{full_prompt}")
        print(f"Expected Answer: {expected}")
        print(f"Predicted Answer: {predicted}")
        print(f"Full Generated Response:\n{response}") 
        print(f"Correct: {'✓' if is_correct else '✗'}")
        print("-"*50)
            
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\n{model_name} Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print("="*50)
    
    model.train()
    return accuracy

def correctness_reward(prompts, completions, answers, **kwargs):
    """根据模型答案的正确性分配奖励。"""
    responses = [comp[0]['content'] for comp in completions] 
    extracted_answers = [extract_answer_from_model_output(r) for r in responses]
    rewards = []
    for pred_ans, expected_ans in zip(extracted_answers, answers):
        current_reward = 0.0
        if pred_ans is not None and expected_ans is not None:
            if str(pred_ans).strip() == str(expected_ans).strip():
                current_reward = 2.0
            else:
                pred_num = _extract_last_number(pred_ans) 
                exp_num = _extract_last_number(expected_ans)
                if pred_num is not None and exp_num is not None and math.isclose(pred_num, exp_num):
                    current_reward = 1.5
                else: 
                    pred_num_single = _extract_single_number(pred_ans)
                    exp_num_single = _extract_single_number(expected_ans)
                    if pred_num_single is not None and exp_num_single is not None and math.isclose(pred_num_single, exp_num_single):
                         current_reward = 1.0 
        rewards.append(current_reward)
    return rewards # 返回 list[float], 长度为 total_batch_size

def format_reward(completions, **kwargs):
    """为遵守期望的XML格式分配奖励。"""
    responses = [comp[0]['content'] for comp in completions]
    rewards = []
    for response in responses:
        score = 0.0
        if "<reasoning>" in response: score += 0.2
        if "</reasoning>" in response: score += 0.2
        if "<answer>" in response: score += 0.2
        if "</answer>" in response: score += 0.2
        rewards.append(score)
    return rewards # 返回 list[float], 长度为 total_batch_size

def combined_reward(prompts, completions, answer):
    """结合正确性和格式奖励。"""
    correctness_scores = correctness_reward(prompts=prompts, completions=completions, answers=answer)
    format_scores = format_reward(completions=completions)
    return [c + f for c, f in zip(correctness_scores, format_scores)] # 返回 list[float], 长度为 total_batch_size

def selective_log_softmax(logits, input_ids):
    """
    计算指定token的对数概率。
    Args:
        logits (torch.Tensor): Shape: (batch_size, seq_len, vocab_size)
        input_ids (torch.Tensor): Shape: (batch_size, seq_len)
    Returns:
        torch.Tensor: Shape: (batch_size, seq_len)
    """
    log_probs = F.log_softmax(logits, dim=-1) # Shape: (batch_size, seq_len, vocab_size)
    selected_log_probs = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)) # Shape: (batch_size, seq_len, 1)
    return selected_log_probs.squeeze(-1) # Shape: (batch_size, seq_len)

def compute_log_probabilities(model, input_ids, attention_mask, completion_len):
    """
    为补全token计算每个token的对数概率。
    Args:
        model: 模型
        input_ids (torch.Tensor): Shape: (batch_size, total_seq_len)
        attention_mask (torch.Tensor): Shape: (batch_size, total_seq_len)
        completion_len (int): 补全部分的长度
    Returns:
        torch.Tensor: Shape: (batch_size, completion_len)
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    logits = outputs.logits # Shape: (batch_size, total_seq_len, vocab_size)

    prompt_len = input_ids.shape[1] - completion_len
    
    pred_logits = logits[:, prompt_len-1:-1, :] # Shape: (batch_size, completion_len, vocab_size)
    target_completion_ids = input_ids[:, prompt_len:] # Shape: (batch_size, completion_len)

    log_probs = selective_log_softmax(pred_logits, target_completion_ids)
    return log_probs # Shape: (batch_size, completion_len)

def create_completion_mask(completion_ids, eos_token_id, pad_token_id):
    """
    为生成的补全token创建二进制掩码，忽略第一个EOS之后的token和pad_token。
    Args:
        completion_ids (torch.Tensor): Shape: (batch_size, seq_len)
        eos_token_id (int): EOS token ID
        pad_token_id (int): PAD token ID
    Returns:
        torch.Tensor: Shape: (batch_size, seq_len)
    """
    mask = torch.ones_like(completion_ids, dtype=torch.bool) # Shape: (batch_size, seq_len)

    for i in range(completion_ids.size(0)):
        eos_indices = (completion_ids[i] == eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_indices) > 0:
            first_eos_idx = eos_indices[0]
            mask[i, first_eos_idx+1:] = 0 
    
    mask[completion_ids == pad_token_id] = 0
    return mask.int() # Shape: (batch_size, seq_len)


def generate_completions_for_model(model, tokenizer, prompts, num_generations, max_completion_length, is_off_policy_model=False):
    """
    为单个模型生成补全。
    Args:
        prompts (list[str]): 如果 is_off_policy_model=True, 则是原始问题列表; 否则是格式化后的提示列表。
        num_generations (int): 每个prompt生成多少个补全。
    Returns:
        prompt_ids_repeated (torch.Tensor): Shape: (batch_size * num_generations, prompt_seq_len)
        prompt_mask_repeated (torch.Tensor): Shape: (batch_size * num_generations, prompt_seq_len)
        completion_ids (torch.Tensor): Shape: (batch_size * num_generations, completion_seq_len)
        completion_attention_mask (torch.Tensor): Shape: (batch_size * num_generations, completion_seq_len) - 考虑了PAD和EOS
    """
    device = next(model.parameters()).device
    tokenizer.padding_side = "left" 

    if is_off_policy_model:
        current_prompts = [build_prompt([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}]) for p in prompts]
    else:
        current_prompts = prompts

    inputs = tokenizer(current_prompts, return_tensors="pt", padding=True).to(device)
    prompt_ids = inputs["input_ids"] # Shape: (actual_batch_size, prompt_seq_len)
    prompt_mask = inputs["attention_mask"] # Shape: (actual_batch_size, prompt_seq_len)
    prompt_length = prompt_ids.size(1)

    prompt_ids_repeated = prompt_ids.repeat_interleave(num_generations, dim=0) # Shape: (actual_batch_size * num_generations, prompt_seq_len)
    prompt_mask_repeated = prompt_mask.repeat_interleave(num_generations, dim=0) # Shape: (actual_batch_size * num_generations, prompt_seq_len)

    outputs = model.generate(
        prompt_ids_repeated,
        attention_mask=prompt_mask_repeated,
        max_new_tokens=max_completion_length,
        do_sample=True,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=1 
    ) # Shape: (actual_batch_size * num_generations, total_seq_len)
    
    completion_ids = outputs[:, prompt_length:] # Shape: (actual_batch_size * num_generations, completion_seq_len)
    
    completion_attention_mask = torch.ones_like(completion_ids, device=device, dtype=torch.long) # Shape: (actual_batch_size*num_generations, completion_seq_len)
    completion_attention_mask[completion_ids == tokenizer.pad_token_id] = 0
    
    for i in range(completion_ids.shape[0]):
        eos_indices = (completion_ids[i] == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_indices) > 0:
            first_eos_idx = eos_indices[0]
            completion_attention_mask[i, first_eos_idx + 1:] = 0

    return prompt_ids_repeated, prompt_mask_repeated, completion_ids, completion_attention_mask


def generate_luffy_rollout_data(policy_model, off_policy_model, ref_model, tokenizer, batch_samples, max_completion_length):
    """
    为LUFFY生成混合策略的rollout数据。
    """
    device = next(policy_model.parameters()).device
    
    prompts_for_on_policy = [sample["prompt"] for sample in batch_samples]
    raw_questions_for_off_policy = [sample["raw_question"] for sample in batch_samples] 
    answers = [sample["answer"] for sample in batch_samples]

    all_prompt_ids_list = []
    all_completion_ids_list = []
    all_completion_masks_list = [] 
    all_input_ids_list = []       
    all_attention_masks_list = [] 
    is_off_policy_flags = []
    old_log_probs_list = []       

    # 1. 在线策略数据生成
    if NUM_ON_POLICY_GENERATIONS > 0:
        # *** CORRECTED LINE: Generate with policy_model ***
        prompt_ids_on, current_prompt_mask_on, completion_ids_on, completion_mask_on = \
            generate_completions_for_model(policy_model, tokenizer, prompts_for_on_policy, NUM_ON_POLICY_GENERATIONS, max_completion_length)
            # prompt_ids_on: Shape: (batch_size * NUM_ON_POLICY_GENERATIONS, prompt_len)
            # current_prompt_mask_on: Shape: (batch_size * NUM_ON_POLICY_GENERATIONS, prompt_len)
            # completion_ids_on: Shape: (batch_size * NUM_ON_POLICY_GENERATIONS, completion_len)
            # completion_mask_on: Shape: (batch_size * NUM_ON_POLICY_GENERATIONS, completion_len)

        input_ids_on = torch.cat([prompt_ids_on, completion_ids_on], dim=1) # Shape: (batch_size * NUM_ON_POLICY_GENERATIONS, total_len)
        attention_mask_on = torch.cat([current_prompt_mask_on, completion_mask_on], dim=1) # Shape: (batch_size*NUM_ON_POLICY_GENERATIONS, total_len)

        with torch.no_grad(): 
            # *** CORRECT: Compute old_log_probs with ref_model ***
            old_log_probs_on = compute_log_probabilities(ref_model, input_ids_on, attention_mask_on, completion_ids_on.size(1))
            # Shape: (batch_size * NUM_ON_POLICY_GENERATIONS, completion_len)

        all_prompt_ids_list.append(prompt_ids_on)
        all_completion_ids_list.append(completion_ids_on)
        all_completion_masks_list.append(completion_mask_on) 
        all_input_ids_list.append(input_ids_on)
        all_attention_masks_list.append(attention_mask_on) 
        old_log_probs_list.extend([p for p in old_log_probs_on])
        is_off_policy_flags.extend([False] * completion_ids_on.size(0))

    # 2. 离策略数据生成
    if NUM_OFF_POLICY_GENERATIONS > 0:
        prompt_ids_off, current_prompt_mask_off, completion_ids_off, completion_mask_off = \
            generate_completions_for_model(off_policy_model, tokenizer, raw_questions_for_off_policy, NUM_OFF_POLICY_GENERATIONS, max_completion_length, is_off_policy_model=True)
            # prompt_ids_off: Shape: (batch_size * NUM_OFF_POLICY_GENERATIONS, prompt_len_off) - prompt_len_off might differ
            # current_prompt_mask_off: Shape: (batch_size * NUM_OFF_POLICY_GENERATIONS, prompt_len_off)
            # completion_ids_off: Shape: (batch_size * NUM_OFF_POLICY_GENERATIONS, completion_len_off)
            # completion_mask_off: Shape: (batch_size * NUM_OFF_POLICY_GENERATIONS, completion_len_off)

        input_ids_off = torch.cat([prompt_ids_off, completion_ids_off], dim=1) # Shape: (batch_size * NUM_OFF_POLICY_GENERATIONS, total_len_off)
        attention_mask_off = torch.cat([current_prompt_mask_off, completion_mask_off], dim=1) # Shape: (batch_size*NUM_OFF_POLICY_GENERATIONS, total_len_off)

        all_prompt_ids_list.append(prompt_ids_off) # This list will contain tensors of potentially different prompt lengths
        all_completion_ids_list.append(completion_ids_off)
        all_completion_masks_list.append(completion_mask_off)
        all_input_ids_list.append(input_ids_off)
        all_attention_masks_list.append(attention_mask_off)
        
        placeholder_shape = old_log_probs_on[0].shape if NUM_ON_POLICY_GENERATIONS > 0 and old_log_probs_on.numel() > 0 else (completion_ids_off.size(1),) # Use completion length of off-policy if on-policy is not available
        num_off_samples = completion_ids_off.size(0)

        if placeholder_shape[0] > 0 : # Check if completion length is greater than 0
             old_log_probs_list.extend([torch.zeros(placeholder_shape, device=device)] * num_off_samples)
        else: # If completion length is 0 (e.g. max_completion_length was 0)
             old_log_probs_list.extend([torch.empty(0, device=device)] * num_off_samples)


        is_off_policy_flags.extend([True] * num_off_samples)

    # 合并所有数据 - 需要处理潜在的序列长度不匹配问题，通常通过填充实现
    # 为了简化，这里假设所有批次的 total_len 和 completion_len 是相同的，或者下游函数能处理变长（通常RL库会填充）
    # 在这个简化版本中，我们依赖于 generate_completions_for_model 内部的填充（如果它做了的话）
    # 或者，更稳健的做法是在这里进行显式填充到所有样本中的最大长度

    final_input_ids = torch.cat(all_input_ids_list, dim=0) # Shape: (total_rollout_batch_size, max_total_len_after_padding)
    final_attention_mask = torch.cat(all_attention_masks_list, dim=0) # Shape: (total_rollout_batch_size, max_total_len_after_padding)
    final_completion_ids = torch.cat(all_completion_ids_list, dim=0) # Shape: (total_rollout_batch_size, max_completion_len_after_padding)
    final_completion_masks = torch.cat(all_completion_masks_list, dim=0) # Shape: (total_rollout_batch_size, max_completion_len_after_padding)
    
    if old_log_probs_list and old_log_probs_list[0].numel() > 0 :
        # 确保所有log_probs张量在堆叠前具有相同的序列长度（补全长度）
        # 如果之前生成时补全长度不一致，这里需要填充
        max_comp_len_in_batch = max(lp.shape[0] for lp in old_log_probs_list if lp.numel() > 0) if any(lp.numel() > 0 for lp in old_log_probs_list) else 0
        
        padded_old_log_probs_list = []
        for lp in old_log_probs_list:
            if lp.numel() == 0: # 处理空张量的情况 (例如，离策略样本的占位符，如果补全长度为0)
                padded_old_log_probs_list.append(torch.zeros((max_comp_len_in_batch,), device=device))
            elif lp.shape[0] < max_comp_len_in_batch:
                padding_size = max_comp_len_in_batch - lp.shape[0]
                # 用0填充log_probs (概率为1，对损失无贡献) 或用一个非常小的值
                padded_lp = F.pad(lp, (0, padding_size), value=0) 
                padded_old_log_probs_list.append(padded_lp)
            else:
                padded_old_log_probs_list.append(lp)
        
        if padded_old_log_probs_list:
            final_old_log_probs = torch.stack(padded_old_log_probs_list, dim=0) # Shape: (total_rollout_batch_size, max_comp_len_in_batch)
        else:
            final_old_log_probs = torch.empty((len(is_off_policy_flags), 0), device=device)
    else:
        final_old_log_probs = torch.empty((len(is_off_policy_flags), final_completion_ids.size(1)), device=device)


    formatted_completions = [
        [{'content': tokenizer.decode(ids, skip_special_tokens=True)}] 
        for ids in final_completion_ids
    ]
    
    repeated_prompts = []
    repeated_answers = []
    num_prompts = len(batch_samples)
    for i in range(num_prompts):
        repeated_prompts.extend([prompts_for_on_policy[i]] * NUM_ON_POLICY_GENERATIONS)
        repeated_answers.extend([answers[i]] * NUM_ON_POLICY_GENERATIONS)
        repeated_prompts.extend([prompts_for_on_policy[i]] * NUM_OFF_POLICY_GENERATIONS)
        repeated_answers.extend([answers[i]] * NUM_OFF_POLICY_GENERATIONS)

    return {
        "input_ids": final_input_ids,             
        "attention_mask": final_attention_mask,   
        "completion_ids": final_completion_ids,   
        "completion_mask": final_completion_masks,
        "old_log_probs": final_old_log_probs,     
        "is_off_policy": torch.tensor(is_off_policy_flags, dtype=torch.bool, device=device), 
        "formatted_completions": formatted_completions, 
        "repeated_prompts": repeated_prompts,     
        "repeated_answers": repeated_answers,     
        "logits_to_keep": final_completion_ids.size(1), 
        "batch_size": len(batch_samples),         
        "num_total_generations_per_prompt": NUM_ON_POLICY_GENERATIONS + NUM_OFF_POLICY_GENERATIONS 
    }


def compute_mixed_group_relative_advantages(rewards, num_total_generations_per_prompt):
    """
    为混合组计算组相对优势值。
    Args:
        rewards (torch.Tensor): Shape: (total_rollout_batch_size,)
        num_total_generations_per_prompt (int): N_on + N_off
    Returns:
        torch.Tensor: Shape: (total_rollout_batch_size,)
    """
    # total_rollout_batch_size = batch_size * num_total_generations_per_prompt
    rewards_by_group = rewards.view(-1, num_total_generations_per_prompt) # Shape: (batch_size, num_total_generations_per_prompt)
    group_means = rewards_by_group.mean(dim=1, keepdim=True) # Shape: (batch_size, 1)
    group_stds = rewards_by_group.std(dim=1, keepdim=True)   # Shape: (batch_size, 1)
    
    expanded_means = group_means.repeat_interleave(num_total_generations_per_prompt, dim=0).squeeze() # Shape: (total_rollout_batch_size,)
    expanded_stds = group_stds.repeat_interleave(num_total_generations_per_prompt, dim=0).squeeze() # Shape: (total_rollout_batch_size,)
    
    advantages = (rewards - expanded_means) / (expanded_stds + 1e-8) # Shape: (total_rollout_batch_size,)
    return advantages


def maximize_luffy_objective(policy_model, rollout_data, tokenizer, reward_function, 
                             optimizer, epsilon_on_policy):
    """通过最大化LUFFY目标来更新策略模型。"""
    device = next(policy_model.parameters()).device

    input_ids = rollout_data["input_ids"].to(device)             # Shape: (total_rollout_batch_size, total_len)
    attention_mask = rollout_data["attention_mask"].to(device)   # Shape: (total_rollout_batch_size, total_len)
    completion_mask = rollout_data["completion_mask"].to(device) # Shape: (total_rollout_batch_size, completion_len) - Mask for completion part
    old_log_probs_from_ref = rollout_data["old_log_probs"].to(device) # Shape: (total_rollout_batch_size, completion_len)
    is_off_policy = rollout_data["is_off_policy"].to(device)     # Shape: (total_rollout_batch_size,)
    logits_to_keep = rollout_data["logits_to_keep"]              # int, completion_len

    current_log_probs = compute_log_probabilities(policy_model, input_ids, attention_mask, logits_to_keep)
    # Shape: (total_rollout_batch_size, completion_len)

    rewards = torch.tensor(
        reward_function(
            prompts=rollout_data["repeated_prompts"], 
            completions=rollout_data["formatted_completions"], 
            answer=rollout_data["repeated_answers"]
        ),
        dtype=torch.float32, device=device
    ) # Shape: (total_rollout_batch_size,)
    avg_reward = rewards.mean().item()
    print(f"Average Reward: {avg_reward:.4f}")

    advantages = compute_mixed_group_relative_advantages(rewards, rollout_data["num_total_generations_per_prompt"])
    # Shape: (total_rollout_batch_size,)
    advantages_expanded = advantages.unsqueeze(1) # Shape: (total_rollout_batch_size, 1)

    total_weighted_loss = torch.tensor(0.0, device=device)
    num_total_samples_in_rollout = input_ids.size(0)
    
    # --- 在线策略损失 ---
    on_policy_selector = ~is_off_policy # Boolean tensor to select on-policy samples
    num_on_policy_samples = on_policy_selector.sum().item()

    if num_on_policy_samples > 0:
        current_log_probs_on = current_log_probs[on_policy_selector] # Shape: (num_on_policy_samples, completion_len)
        
        # Ensure old_log_probs_from_ref is valid and correctly shaped before indexing
        if old_log_probs_from_ref.numel() > 0 and old_log_probs_from_ref.shape[0] == num_total_samples_in_rollout and old_log_probs_from_ref.shape[1] == logits_to_keep:
            old_log_probs_on = old_log_probs_from_ref[on_policy_selector] # Shape: (num_on_policy_samples, completion_len)
        else:
            print(f"Warning: Skipping on-policy loss. old_log_probs_from_ref shape: {old_log_probs_from_ref.shape}, expected first dim: {num_total_samples_in_rollout}, second dim: {logits_to_keep}")
            old_log_probs_on = None # Mark as invalid

        if old_log_probs_on is not None and old_log_probs_on.numel() > 0 :
            advantages_on = advantages_expanded[on_policy_selector]         # Shape: (num_on_policy_samples, 1)
            completion_mask_on = completion_mask[on_policy_selector]       # Shape: (num_on_policy_samples, completion_len)

            ratio_on = torch.exp(current_log_probs_on - old_log_probs_on)      # Shape: (num_on_policy_samples, completion_len)
            surrogate1_on = ratio_on * advantages_on                           # Shape: (num_on_policy_samples, completion_len)
            surrogate2_on = torch.clamp(ratio_on, 1 - epsilon_on_policy, 1 + epsilon_on_policy) * advantages_on # Shape: (num_on_policy_samples, completion_len)
            
            per_token_loss_on = torch.min(surrogate1_on, surrogate2_on)        # Shape: (num_on_policy_samples, completion_len)
            
            masked_sum_loss_on = (per_token_loss_on * completion_mask_on).sum(dim=1) # Shape: (num_on_policy_samples,)
            num_valid_tokens_on = completion_mask_on.sum(dim=1).clamp(min=1)   # Shape: (num_on_policy_samples,)
            avg_seq_loss_on = (masked_sum_loss_on / num_valid_tokens_on).mean() # Scalar
            total_weighted_loss += avg_seq_loss_on * num_on_policy_samples 
            print(f"  On-policy Loss part (avg per seq): {avg_seq_loss_on.item():.4f} (from {num_on_policy_samples} samples)")
        elif num_on_policy_samples > 0 : # num_on_policy_samples > 0 but old_log_probs_on is None or empty
             print(f"  Skipping On-policy Loss part due to missing old_log_probs for {num_on_policy_samples} samples.")


    # --- 离策略损失 ---
    off_policy_selector = is_off_policy # Boolean tensor
    num_off_policy_samples = off_policy_selector.sum().item()

    if num_off_policy_samples > 0:
        current_log_probs_off = current_log_probs[off_policy_selector] # Shape: (num_off_policy_samples, completion_len)
        advantages_off = advantages_expanded[off_policy_selector]       # Shape: (num_off_policy_samples, 1)
        completion_mask_off = completion_mask[off_policy_selector]     # Shape: (num_off_policy_samples, completion_len)

        pi_theta_token_probs_off = torch.exp(current_log_probs_off)        # Shape: (num_off_policy_samples, completion_len)
        
        shaped_ratio_off = pi_theta_token_probs_off / (pi_theta_token_probs_off + GAMMA_POLICY_SHAPING) # Shape: (num_off_policy_samples, completion_len)
        
        per_token_loss_off = shaped_ratio_off * advantages_off             # Shape: (num_off_policy_samples, completion_len)
        
        masked_sum_loss_off = (per_token_loss_off * completion_mask_off).sum(dim=1) # Shape: (num_off_policy_samples,)
        num_valid_tokens_off = completion_mask_off.sum(dim=1).clamp(min=1) # Shape: (num_off_policy_samples,)
        avg_seq_loss_off = (masked_sum_loss_off / num_valid_tokens_off).mean() # Scalar
        total_weighted_loss += avg_seq_loss_off * num_off_policy_samples 
        print(f"  Off-policy Loss part (avg per seq): {avg_seq_loss_off.item():.4f} (from {num_off_policy_samples} samples)")

    if num_total_samples_in_rollout > 0:
        final_loss = - (total_weighted_loss / num_total_samples_in_rollout) # Scalar
    else:
        final_loss = torch.tensor(0.0, device=device, requires_grad=True) 

    optimizer.zero_grad()
    if final_loss.requires_grad: 
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0) 
        optimizer.step()
    else:
        print("Warning: Final loss does not require grad. Skipping backward and step.")
    
    return final_loss.item()


def train_with_luffy(policy_model, off_policy_model, tokenizer, train_data, 
                     num_iterations=1, steps_per_iteration=100, 
                     batch_size=2, 
                     max_completion_length=128, 
                     learning_rate=5e-6, 
                     mu_updates_per_batch=1, 
                     epsilon_on_policy=0.2, 
                     reward_function=combined_reward,
                     eval_data=None, 
                     eval_freq=25   
                     ):
    """使用LUFFY算法进行迭代训练。"""
    device = next(policy_model.parameters()).device
    
    for iteration in range(1, num_iterations + 1):
        print(f"\n--- Starting LUFFY Iteration {iteration}/{num_iterations} ---")
        
        ref_model = copy.deepcopy(policy_model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model = ref_model.to(device)
        
        off_policy_model.eval()
        for param in off_policy_model.parameters():
            param.requires_grad = False
        off_policy_model = off_policy_model.to(device)

        optimizer = torch.optim.Adam(policy_model.parameters(), lr=learning_rate)
        policy_model.train()
        
        for step in range(1, steps_per_iteration + 1):
            if len(train_data) < batch_size:
                 print(f"Warning: Not enough training data ({len(train_data)}) for batch size ({batch_size}). Skipping step {step} in iteration {iteration}.")
                 continue 
            
            batch_samples = random.sample(train_data, batch_size)
            
            with torch.no_grad(): 
                rollout_data = generate_luffy_rollout_data(
                    policy_model, off_policy_model, ref_model, tokenizer, 
                    batch_samples, max_completion_length
                )
            
            if rollout_data["input_ids"].size(0) == 0:
                 print(f"Warning: No rollout data generated for step {step}, iteration {iteration}. Skipping update.")
                 continue

            for update_iter in range(1, mu_updates_per_batch + 1):
                loss_value = maximize_luffy_objective(
                    policy_model, rollout_data, tokenizer,
                    reward_function, optimizer, epsilon_on_policy
                )
                print(f"Iter {iteration}, Step {step}/{steps_per_iteration}, "
                      f"Update {update_iter}/{mu_updates_per_batch}, Loss: {loss_value:.4f}")

            if eval_data and step % eval_freq == 0:
                print(f"\n--- Evaluating at Iter {iteration}, Step {step} ---")
                evaluate_model(policy_model, tokenizer, eval_data, device, model_name="PolicyModel (Intermediate)")
                policy_model.train() 

        print(f"--- Completed LUFFY Iteration {iteration} ---")
    
    return policy_model

def optimize_model_memory(model):
    """应用内存优化，如正确的梯度检查点设置。"""
    model.train()
    model.config.use_cache = False
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    on_policy_model_name = "Qwen/Qwen2-0.5B-Instruct" 
    off_policy_model_name = "deepseek-ai/deepseek-coder-1.3b-instruct" 
    output_dir = "luffy_math_solver_model_corrected" # Changed output directory

    print(f"Loading on-policy model ({on_policy_model_name})...")
    policy_model = AutoModelForCausalLM.from_pretrained(
        on_policy_model_name,
        torch_dtype=torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32,
        trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(on_policy_model_name, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        print("Tokenizer does not have a pad token, setting it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    policy_model.config.pad_token_id = tokenizer.pad_token_id
    
    print(f"Loading off-policy model ({off_policy_model_name})...")
    off_policy_model = AutoModelForCausalLM.from_pretrained(
        off_policy_model_name,
        torch_dtype=torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32,
        trust_remote_code=True
    ).to(device)
    
    print("Preparing dataset...")
    all_data = prepare_dataset("train")
    random.shuffle(all_data)
    
    num_eval_examples = min(10, max(1, len(all_data) // 10 if len(all_data) >= 20 else 1)) if all_data else 0
    
    if not all_data or len(all_data) < 2 : # Need at least one for train and one for eval if possible
        print("Error: Dataset is too small or empty. Exiting.")
        return

    if len(all_data) <= num_eval_examples: 
        print(f"Warning: Dataset size ({len(all_data)}) is small. Using all data for training and {min(1, len(all_data))} for eval if possible, or skipping eval.")
        train_data = all_data
        if len(all_data) > 0:
            eval_data = all_data[:min(1, len(all_data))] # Use at least one sample for eval if available
            num_eval_examples = len(eval_data)
        else:
            eval_data = None
            num_eval_examples = 0
    else:
        eval_data = all_data[:num_eval_examples]
        train_data = all_data[num_eval_examples:]

    print(f"Using {len(train_data)} samples for training and {num_eval_examples} for evaluation.")

    if eval_data:
        print("\nInitial on-policy model evaluation:")
        evaluate_model(policy_model, tokenizer, eval_data, device, model_name="On-Policy Model (Initial)")
        print("\nInitial off-policy model evaluation (for reference):")
        evaluate_model(off_policy_model, tokenizer, eval_data, device, model_name="Off-Policy Model (Reference)")
    
    print("\nStarting LUFFY finetuning...")
    training_config = {
        'num_iterations': 1, 
        'steps_per_iteration': 5, # Further reduced for quick testing
        'batch_size': 1,          # Reduced for quick testing
        'max_completion_length': 50, 
        'learning_rate': 1e-6, 
        'mu_updates_per_batch': 1, 
        'epsilon_on_policy': 0.2, 
        'reward_function': combined_reward,
        'eval_data': eval_data,
        'eval_freq': 2 
    }
    if not train_data:
        print("Error: No training data available. Skipping training.")
    else:
        policy_model = train_with_luffy(
            policy_model=policy_model,
            off_policy_model=off_policy_model,
            tokenizer=tokenizer,
            train_data=train_data,
            **training_config
        )

    if eval_data:
        print("\nFinal on-policy model evaluation after LUFFY finetuning:")
        evaluate_model(policy_model, tokenizer, eval_data, device, model_name="On-Policy Model (Finetuned with LUFFY)")

    print(f"\nSaving LUFFY finetuned model to {output_dir}...")
    policy_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model saving complete.")

if __name__ == "__main__":
    main()
