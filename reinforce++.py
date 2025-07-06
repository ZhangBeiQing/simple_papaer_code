
"""
### 核心改动总结

1.  **优势函数（Advantage Function）是唯一的核心区别**：
    *   **GRPO**: 为每个prompt生成一组（Group）回答，然后用**组内的平均分和标准差**来归一化奖励，得到“相对优势”。我保留了原来的`compute_group_relative_advantages`函数。
    *   **REINFORCE++**: 为每个prompt只生成一个回答，然后用**整个训练批次（Global Batch）的平均分和标准差**来归一化奖励，得到“全局优势”。为此，我新增了一个函数`compute_reinforce_pp_advantages`。

2.  **生成数量（Number of Generations）不同**：
    *   GRPO的精髓在于“组内比较”，所以它必须为每个prompt生成多个回答 (`num_generations > 1`)。
    *   REINFORCE++的核心在于“全局比较”，所以它最标准的实现是为每个prompt只生成一个回答 (`num_generations = 1`)。

3.  **代码结构**：
    *   我新增了`train_with_reinforce_pp`和`maximize_reinforce_pp_objective`函数，它们是`train_with_grpo`和`maximize_grpo_objective`的“镜像版本”，绝大部分代码都相同，**唯一的区别就是调用了不同的优势函数计算方法**。
    *   为了方便您运行和比较，我在`main`函数中加入了命令行参数`--algorithm`，您可以通过 `--algorithm reinforce_pp` 来运行新的实现。

下面，我将为您展示新增的核心代码片段，并在最后提供完整的、可运行的文件。

---

### 新增的核心代码解读

#### 1. REINFORCE++ 的优势函数计算

这是最关键的新增部分。注意它的计算方式与GRPO的`compute_group_relative_advantages`有何不同。

```python
def compute_reinforce_pp_advantages(rewards):

    Compute REINFORCE++ advantages using global batch normalization.
    This is the core difference from GRPO's group-relative approach.
    
    Args:
        rewards (torch.Tensor): A flat tensor of shape (total_batch_size) containing rewards
                                for every generated completion in the batch.
        
    Returns:
        torch.Tensor: Tensor of advantages computed relative to the global batch mean and std.

    # Compute mean and standard deviation across the ENTIRE batch
    global_mean = rewards.mean()
    global_std = rewards.std()
    
    # Normalize rewards using the global batch statistics
    advantages = (rewards - global_mean) / (global_std + 1e-8) # Use a small epsilon for stability
    
    # Add a dimension for token-wise multiplication later
    return advantages.unsqueeze(1)
```

**解读**：这个函数非常简单。它接收整个批次的所有奖励，计算全局的均值和标准差，然后进行归一化。这与GRPO对每个小组（prompt）分别计算均值和标准差形成了鲜明对比。

#### 2. REINFORCE++ 的优化目标函数

这个函数几乎是`maximize_grpo_objective`的翻版，唯一改变的就是优势函数的调用。

```python
def maximize_reinforce_pp_objective(model, ref_model, rollout_data, tokenizer, reward_function, 
                                  optimizer, beta, epsilon):

    Update the policy model by maximizing the REINFORCE++ objective.
    This is nearly identical to the GRPO version, except for the advantage calculation.

    # ... (代码与GRPO版本完全相同，直到计算rewards) ...
    
    # Compute rewards
    rewards = torch.tensor(
        reward_function(prompts=repeated_prompts, completions=formatted_completions, answer=repeated_answers),
        dtype=torch.float32,
        device=next(model.parameters()).device
    )
    avg_reward = rewards.mean().item()
    print(f"Global Average Reward: {avg_reward:.4f}")
    
    # *** THE CORE CHANGE IS HERE ***
    # Compute advantages using global batch normalization
    advantages = compute_reinforce_pp_advantages(rewards)
    
    # ... (剩余代码，包括surrogate_loss, kl_div, loss计算和优化步骤，与GRPO版本完全相同) ...
    
    # ...
    return loss.item()

```
**解读**：这清晰地表明，REINFORCE++和GRPO都共享PPO的代理目标（Clipping Surrogate Objective）和KL散度惩罚项。它们的**唯一区别**就是如何定义和计算“优势（Advantage）”。

---
"""
# filename: GRPO_and_REINFORCE_PP.py

import numpy as np
import random
import torch
import torch.nn.functional as F
import copy
import argparse # Added for algorithm selection

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def set_random_seed(seed: int = 42):
    """
    Set the random seed for reproducibility across Python, NumPy, and PyTorch.

    Parameters:
        seed (int): The seed value to use.
    """
    # Set the seed for Python's built-in random module
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)

    # Set the seed for PyTorch
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN (may impact performance)
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
    """Load and prepare the GSM8K dataset for training with string prompts."""
    data = load_dataset('openai/gsm8k', 'main')[split]
    formatted_data = []

    for example in data:
        # Convert list of messages to a single string prompt.
        prompt_str = build_prompt([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]}
        ])
        formatted_example = {
            "prompt": prompt_str,  # Now a string rather than a list.
            "answer": extract_answer_from_dataset(example["answer"])
        }
        formatted_data.append(formatted_example)

    return formatted_data

def build_prompt(messages):
    """
    Build a single prompt string from a list of messages.
    Each message is expected to be a dictionary with 'role' and 'content' keys.
    This function concatenates all message contents, preserving the training format.
    """
    return "\n".join([msg["content"].strip() for msg in messages])

def extract_answer_from_model_output(text):
    """
    Extracts the value from the last <answer> tag in the text.
    Returns None if no valid answer is found.
    """
    # Split on <answer> and take everything after the last occurrence
    parts = text.split("<answer>")
    if len(parts) < 2:  # No <answer> tag found
        return None

    last_part = parts[-1]

    # Extract content up to </answer>
    if "</answer>" not in last_part:
        return None

    answer = last_part.split("</answer>")[0].strip()
    return None if answer == "..." else answer

def extract_answer_from_dataset(text):
    """
    Extracts the answer from the dataset.
    The dataset separates the answer using the '####' delimiter.
    """
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def _extract_last_number(text):
    """
    Extracts the last number from text if it's properly separated.
    
    Args:
        text (str): The text to extract a number from.
        
    Returns:
        float or None: The extracted number as a float, or None if no valid number is found.
    """
    import re
    text = text.replace('$', '').replace('%', '')
    pattern = r'(?:^|\s|=)\s*(-?\d*\.?\d+)\s*$'
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def _extract_single_number(text):
    """
    Extracts a single number from text if exactly one exists.
    
    Args:
        text (str): The text to extract a number from.
        
    Returns:
        float or None: The extracted number as a float if exactly one number exists, otherwise None.
    """
    import re
    numbers = re.findall(r'-?\d*\.?\d+', text)
    return float(numbers[0]) if len(numbers) == 1 else None


def evaluate_model(model, tokenizer, eval_examples, device):
    """
    Evaluates the model on a set of examples and prints detailed results.
    """
    model.eval()
    correct = 0
    total = len(eval_examples)
    print("\n" + "="*50)
    print("EVALUATION ON", total, "EXAMPLES")
    print("="*50)
    
    for example in eval_examples:
        full_prompt = example["prompt"]
        expected = example["answer"]
        
        inputs = tokenizer.encode(full_prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            inputs,
            max_new_tokens=512,
            temperature=0.7,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            forced_eos_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        try:
            predicted = extract_answer_from_model_output(response)
            
            is_correct = False
            if predicted == expected:
                is_correct = True
            else:
                pred_num = _extract_single_number(str(predicted))
                exp_num = _extract_single_number(str(expected))
                if pred_num is not None and exp_num is not None and pred_num == exp_num:
                    is_correct = True
                else:
                    pred_num = _extract_last_number(str(predicted))
                    exp_num = _extract_last_number(str(expected))
                    is_correct = (pred_num is not None and exp_num is not None and
                                pred_num == exp_num)

            if is_correct:
                correct += 1
                
            print("\nPrompt:")
            print(full_prompt)
            print("\nExpected Answer:")
            print(expected)
            print("\nExtracted Answer:")
            print(predicted)
            print("\nFull Generated Response:")
            print(response)
            print("\nCorrect:", "✓" if is_correct else "✗")
            print("-"*50)
            
        except Exception as e:
            print(f"\nFailed to parse model output for prompt:\n{full_prompt}\nError: {e}")
            print("-"*50)
            
    accuracy = (correct / total) * 100
    print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")
    print("="*50)
    
    model.train()
    return accuracy

def correctness_reward(prompts, completions, answer, **kwargs):
    """
    Assigns a reward based on the correctness of the model's answer.
    """
    responses = [completion[0]['content'] for completion in completions]
    extracted = [extract_answer_from_model_output(r) for r in responses]

    rewards = []
    for r, a in zip(extracted, answer):
        if r == a:
            rewards.append(2.0)
        else:
            r_num = _extract_single_number(str(r))
            a_num = _extract_single_number(str(a))
            if r_num is not None and a_num is not None and r_num == a_num:
                rewards.append(1.5)
            else:
                rewards.append(0.0)
    return rewards

def format_reward(completions, **kwargs):
    """
    Assigns a reward for adhering to the desired XML format.
    """
    responses = [completion[0]['content'] for completion in completions]
    rewards = []
    for response in responses:
        score = 0.0
        if "<reasoning>" in response: score += 0.2
        if "</reasoning>" in response: score += 0.2
        if "<answer>" in response: score += 0.2
        if "</answer>" in response: score += 0.2
        rewards.append(score)
    return rewards

def combined_reward(prompts, completions, answer):
    """
    Combines correctness and format rewards.
    """
    correctness_scores = correctness_reward(prompts=prompts, completions=completions, answer=answer)
    format_scores = format_reward(completions=completions)
    return [c + f for c, f in zip(correctness_scores, format_scores)]

def selective_log_softmax(logits, input_ids):
    """
    Compute the log probabilities for the tokens specified in input_ids.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    selected_log_probs = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1))
    return selected_log_probs.squeeze(-1)

def compute_log_probabilities(model, input_ids, attention_mask, logits_to_keep):
    """
    Compute per-token log probabilities for a subset of tokens.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits = outputs.logits

    logits = logits[:, :-1, :]
    input_ids_for_log_probs = input_ids[:, 1:]

    log_probs = selective_log_softmax(logits, input_ids_for_log_probs)

    # We only care about the log_probs of the completion part
    return log_probs[:, -logits_to_keep:]


def create_completion_mask(completion_ids, eos_token_id):
    """
    Create a binary mask for the generated completion tokens.
    """
    is_eos = completion_ids == eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
    mask_exists = is_eos.any(dim=1)
    if mask_exists.any():
        eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]
    sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
    return (sequence_indices <= eos_idx.unsqueeze(1)).int()

def generate_completions(model, tokenizer, prompts, num_generations=4, max_completion_length=32):
    """
    Generate multiple completions for each prompt.
    """
    device = next(model.parameters()).device
    tokenizer.padding_side  = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    prompt_ids = inputs["input_ids"].to(device)
    prompt_mask = inputs["attention_mask"].to(device)
    prompt_length = prompt_ids.size(1)

    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)

    outputs = model.generate(
        prompt_ids,
        attention_mask=prompt_mask,
        max_new_tokens=max_completion_length,
        do_sample=True,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    completion_ids = outputs[:, prompt_length:]
    completion_mask = create_completion_mask(completion_ids, tokenizer.eos_token_id)

    return prompt_ids, prompt_mask, completion_ids, completion_mask

def generate_rollout_data(model, ref_model, tokenizer, batch_samples, num_generations, max_completion_length):
    """
    Generate rollouts and compute static log probabilities.
    """
    prompts = [sample["prompt"] for sample in batch_samples]
    answers = [sample["answer"] for sample in batch_samples]

    with torch.no_grad():
        prompt_ids, prompt_mask, completion_ids, completion_mask = generate_completions(
            model, tokenizer, prompts, num_generations, max_completion_length
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        old_log_probs = compute_log_probabilities(model, input_ids, attention_mask, logits_to_keep)
        ref_log_probs = compute_log_probabilities(ref_model, input_ids, attention_mask, logits_to_keep)

    formatted_completions = [
        [{'content': tokenizer.decode(ids, skip_special_tokens=True)}]
        for ids in completion_ids
    ]
    repeated_prompts = [p for p in prompts for _ in range(num_generations)]
    repeated_answers = [a for a in answers for _ in range(num_generations)]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": completion_mask,
        "old_log_probs": old_log_probs,
        "ref_log_probs": ref_log_probs,
        "formatted_completions": formatted_completions,
        "repeated_prompts": repeated_prompts,
        "repeated_answers": repeated_answers,
        "logits_to_keep": logits_to_keep,
        "batch_size": len(prompts),
        "num_generations": num_generations
    }

# ===================================================================
# GRPO SPECIFIC FUNCTIONS
# ===================================================================
def compute_group_relative_advantages(rewards, num_generations):
    """
    Compute group-relative advantages for each prompt group (GRPO).
    """
    rewards_by_group = rewards.view(-1, num_generations)
    group_means = rewards_by_group.mean(dim=1)
    group_stds = rewards_by_group.std(dim=1)
    
    expanded_means = group_means.repeat_interleave(num_generations)
    expanded_stds = group_stds.repeat_interleave(num_generations)
    
    advantages = (rewards - expanded_means) / (expanded_stds + 1e-8)
    return advantages.unsqueeze(1)

def maximize_grpo_objective(model, ref_model, rollout_data, tokenizer, reward_function, 
                          optimizer, beta, epsilon):
    """
    Update the policy model by maximizing the GRPO objective.
    """
    input_ids = rollout_data["input_ids"]
    attention_mask = rollout_data["attention_mask"]
    completion_mask = rollout_data["completion_mask"]
    old_log_probs = rollout_data["old_log_probs"]
    ref_log_probs = rollout_data["ref_log_probs"]
    logits_to_keep = rollout_data["logits_to_keep"]
    
    current_log_probs = compute_log_probabilities(model, input_ids, attention_mask, logits_to_keep)
    ratio = torch.exp(current_log_probs - old_log_probs)
    
    rewards = torch.tensor(
        reward_function(prompts=rollout_data["repeated_prompts"], completions=rollout_data["formatted_completions"], answer=rollout_data["repeated_answers"]),
        dtype=torch.float32,
        device=next(model.parameters()).device
    )
    avg_reward = rewards.mean().item()
    print(f"Average Reward: {avg_reward:.4f}")
    
    advantages = compute_group_relative_advantages(rewards, rollout_data["num_generations"])
    
    surrogate1 = ratio * advantages
    surrogate2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    surrogate_loss = torch.min(surrogate1, surrogate2)
    
    # Using the k3 KL divergence approximation from the paper for consistency
    kl_div = torch.exp(ref_log_probs - current_log_probs) - (ref_log_probs - current_log_probs) - 1
    
    per_token_loss = surrogate_loss - beta * kl_div
    loss = -((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
    optimizer.step()
    
    return loss.item()

# ===================================================================
# REINFORCE++ SPECIFIC FUNCTIONS (NEWLY ADDED)
# ===================================================================
def compute_reinforce_pp_advantages(rewards):
    """
    Compute REINFORCE++ advantages using global batch normalization.
    This is the core difference from GRPO's group-relative approach.
    
    Args:
        rewards (torch.Tensor): A flat tensor of shape (total_batch_size) containing rewards
                                for every generated completion in the batch.
        
    Returns:
        torch.Tensor: Tensor of advantages computed relative to the global batch mean and std.
    """
    # Compute mean and standard deviation across the ENTIRE batch
    global_mean = rewards.mean()
    global_std = rewards.std()
    
    # Normalize rewards using the global batch statistics
    advantages = (rewards - global_mean) / (global_std + 1e-8) # Use a small epsilon for stability
    
    # Add a dimension for token-wise multiplication later
    return advantages.unsqueeze(1)

def maximize_reinforce_pp_objective(model, ref_model, rollout_data, tokenizer, reward_function, 
                                  optimizer, beta, epsilon):
    """
    Update the policy model by maximizing the REINFORCE++ objective.
    This is nearly identical to the GRPO version, except for the advantage calculation.
    """
    input_ids = rollout_data["input_ids"]
    attention_mask = rollout_data["attention_mask"]
    completion_mask = rollout_data["completion_mask"]
    old_log_probs = rollout_data["old_log_probs"]
    ref_log_probs = rollout_data["ref_log_probs"]
    logits_to_keep = rollout_data["logits_to_keep"]
    
    current_log_probs = compute_log_probabilities(model, input_ids, attention_mask, logits_to_keep)
    ratio = torch.exp(current_log_probs - old_log_probs)
    
    rewards = torch.tensor(
        reward_function(prompts=rollout_data["repeated_prompts"], completions=rollout_data["formatted_completions"], answer=rollout_data["repeated_answers"]),
        dtype=torch.float32,
        device=next(model.parameters()).device
    )
    avg_reward = rewards.mean().item()
    print(f"Global Average Reward: {avg_reward:.4f}")
    
    # *** THE CORE CHANGE IS HERE ***
    # Compute advantages using global batch normalization
    advantages = compute_reinforce_pp_advantages(rewards)
    
    # The rest of the objective function is identical to PPO/GRPO
    surrogate1 = ratio * advantages
    surrogate2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    surrogate_loss = torch.min(surrogate1, surrogate2)
    
    kl_div = torch.exp(ref_log_probs - current_log_probs) - (ref_log_probs - current_log_probs) - 1
    
    per_token_loss = surrogate_loss - beta * kl_div
    loss = -((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
    optimizer.step()
    
    return loss.item()

# ===================================================================
# TRAINING LOOPS
# ===================================================================
def train_with_grpo(model, tokenizer, train_data, num_iterations=1, 
                           steps_per_iteration=500, batch_size=4, num_generations=4, 
                           max_completion_length=128, beta=0.1, learning_rate=5e-6, 
                           mu=3, epsilon=0.2, reward_function=combined_reward):
    """
    Iterative Group Relative Policy Optimization algorithm.
    """
    policy_model = model
    device = next(policy_model.parameters()).device
    
    for iteration in range(1, num_iterations + 1):
        print(f"\nStarting GRPO iteration {iteration}/{num_iterations}")
        
        reference_model = copy.deepcopy(policy_model)
        reference_model.eval()
        for param in reference_model.parameters(): param.requires_grad = False
        
        optimizer = torch.optim.Adam(policy_model.parameters(), lr=learning_rate)
        policy_model.train()
        
        for step in range(1, steps_per_iteration + 1):
            batch_samples = random.sample(train_data, batch_size)
            
            with torch.no_grad():
                rollout_data = generate_rollout_data(
                    policy_model, reference_model, tokenizer, 
                    batch_samples, num_generations, max_completion_length
                )
            
            for grpo_iter in range(1, mu + 1):
                loss_value = maximize_grpo_objective(
                    policy_model, reference_model, rollout_data, tokenizer,
                    reward_function, optimizer, beta, epsilon
                )
                print(f"Iter {iteration}, Step {step}/{steps_per_iteration}, "
                      f"GRPO update {grpo_iter}/{mu}, Loss: {loss_value:.4f}")
    
    return policy_model

def train_with_reinforce_pp(model, tokenizer, train_data, num_iterations=1, 
                           steps_per_iteration=500, batch_size=4,
                           max_completion_length=128, beta=0.1, learning_rate=5e-6, 
                           mu=3, epsilon=0.2, reward_function=combined_reward):
    """
    REINFORCE++ training loop. Note that num_generations is fixed to 1.
    """
    policy_model = model
    device = next(policy_model.parameters()).device
    
    # REINFORCE++ uses 1 generation per prompt
    num_generations = 1

    for iteration in range(1, num_iterations + 1):
        print(f"\nStarting REINFORCE++ iteration {iteration}/{num_iterations}")
        
        reference_model = copy.deepcopy(policy_model)
        reference_model.eval()
        for param in reference_model.parameters(): param.requires_grad = False
        
        optimizer = torch.optim.Adam(policy_model.parameters(), lr=learning_rate)
        policy_model.train()
        
        for step in range(1, steps_per_iteration + 1):
            batch_samples = random.sample(train_data, batch_size)
            
            with torch.no_grad():
                rollout_data = generate_rollout_data(
                    policy_model, reference_model, tokenizer, 
                    batch_samples, num_generations, max_completion_length
                )
            
            for pp_iter in range(1, mu + 1):
                loss_value = maximize_reinforce_pp_objective(
                    policy_model, reference_model, rollout_data, tokenizer,
                    reward_function, optimizer, beta, epsilon
                )
                print(f"Iter {iteration}, Step {step}/{steps_per_iteration}, "
                      f"REINFORCE++ update {pp_iter}/{mu}, Loss: {loss_value:.4f}")
    
    return policy_model

def optimize_model_memory(model):
    """Apply memory optimizations."""
    model.train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    return model

def main():
    """
    Main function to run the complete training and evaluation pipeline.
    """
    # ADDED: Argument Parser to select algorithm
    parser = argparse.ArgumentParser(description="Finetune a model with GRPO or REINFORCE++.")
    parser.add_argument(
        '--algorithm', 
        type=str, 
        default='grpo', 
        choices=['grpo', 'reinforce_pp'],
        help='The RL algorithm to use for finetuning.'
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir = f"{args.algorithm}_finetuned_model"

    print("Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=None
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    all_data = prepare_dataset("train")
    random.shuffle(all_data)
    num_eval_examples = 5
    eval_data = all_data[:num_eval_examples]
    train_data = all_data[num_eval_examples:]

    print(f"\nInitial model evaluation before {args.algorithm.upper()}:")
    pre_accuracy = evaluate_model(model, tokenizer, eval_data, device)
    print(f"Pre-Finetuning Accuracy: {pre_accuracy:.2f}%")

    model = optimize_model_memory(model)
    
    print(f"\nStarting RL finetuning using {args.algorithm.upper()}...")

    # Select and run the chosen algorithm
    if args.algorithm == 'grpo':
        training_config = {
            'num_iterations' : 1,
            'steps_per_iteration': 50,
            'batch_size': 2,
            'num_generations': 4, # GRPO needs > 1
            'max_completion_length': 256,
            'beta': 0.04,
            'learning_rate': 5e-7,
            'mu': 1,
            'epsilon': 0.1,
            'reward_function': combined_reward
        }
        model = train_with_grpo(model, tokenizer, train_data, **training_config)
    
    elif args.algorithm == 'reinforce_pp':
        training_config = {
            'num_iterations' : 1,
            'steps_per_iteration': 50,
            'batch_size': 8, # Can use larger batch size as num_generations=1
            'max_completion_length': 256,
            'beta': 0.04,
            'learning_rate': 5e-7,
            'mu': 1,
            'epsilon': 0.1,
            'reward_function': combined_reward
        }
        model = train_with_reinforce_pp(model, tokenizer, train_data, **training_config)

    print(f"\nFinal model evaluation after {args.algorithm.upper()} RL finetuning:")
    post_accuracy = evaluate_model(model, tokenizer, eval_data, device)
    print(f"Post-{args.algorithm.upper()} Accuracy: {post_accuracy:.2f}%")
    print(f"Total Improvement: {post_accuracy - pre_accuracy:.2f}%")

    print(f"\nSaving {args.algorithm.upper()} finetuned model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()