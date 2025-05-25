import random
import time
import ast # For basic syntax checking
from copy import deepcopy
from collections import defaultdict
import re

# --- Hugging Face Transformers ---
# 你可能需要先安装: pip install transformers torch sentencepiece
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# --- 修改后的模拟组件 ---

class RealLLM:
    """
    使用Hugging Face基础模型模拟LLM。
    """
    def __init__(self, model_name="gpt2"): # 可以换成 "distilgpt2" 更小
        print(f"Loading Hugging Face model: {model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            # 设置pad_token_id，如果模型没有的话
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                self.model.config.pad_token_id = self.model.config.eos_token_id
            print(f"Model {model_name} loaded successfully.")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            print("Please ensure you have an internet connection and the model name is correct.")
            print("You might need to install 'transformers' and 'torch'.")
            raise

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model_state_counter = 0 # 用于模拟模型被更新

    def generate_text(self, prompt: str, task_type: str, role: str, max_new_tokens=50) -> str:
        """
        根据提示、任务类型和角色生成文本。
        """
        # print(f"\n--- LLM Received Prompt ({role}, {task_type}) ---\n{prompt[:300]}...\n--- END LLM Prompt ---")

        # 为了简化，我们让模型总是尝试生成符合某种格式的输出
        # 实际应用中，提示工程会更精细
        if role == "proposer":
            if task_type == "deduction" or task_type == "abduction":
                # 期望生成 ```python\n...\n```\n```input\n...\n```
                # 我们在prompt末尾添加引导词
                extended_prompt = prompt + "\n```python\ndef f(x):\n  return" # 引导模型生成代码
            elif task_type == "induction":
                # 期望生成 ```input\n...\n```\n```message\n...\n```
                extended_prompt = prompt + "\n```input\n1" # 引导模型生成输入
            else:
                extended_prompt = prompt
        elif role == "solver":
            if task_type == "induction":
                extended_prompt = prompt + "\n```python\ndef f(x):\n  return" # 引导模型生成代码
            else:
                extended_prompt = prompt + "\n" # 引导模型生成答案
        else:
            extended_prompt = prompt

        inputs = self.tokenizer(extended_prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        # 注意：这里的生成参数非常简化
        # 真实的AZR会使用更复杂的采样策略和更长的生成长度
        # 这里的 max_new_tokens 较小，是为了演示速度
        # 对于代码生成，可能需要更大的 max_new_tokens
        if task_type == "induction" and role == "solver": # 生成程序
            actual_max_new_tokens = 100
        elif task_type in ["deduction", "abduction"] and role == "proposer": # 生成程序和输入
            actual_max_new_tokens = 80
        else:
            actual_max_new_tokens = max_new_tokens

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=actual_max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=True, # 稍微增加点随机性
                    top_k=10,
                    temperature=0.8
                )
            # 解码时跳过特殊token，也跳过输入提示部分
            generated_text = self.tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        except Exception as e:
            # print(f"Error during LLM generation: {e}")
            generated_text = "Error: LLM generation failed."

        # print(f"--- LLM Generated Text ---\n{generated_text[:300]}...\n--- END LLM Generated Text ---")
        return generated_text

    def update(self, rewards: dict):
        # 模拟模型更新
        self.model_state_counter += 1
        # print(f"LLM state counter: {self.model_state_counter}, last rewards: {rewards}")
        # 实际中，这里会使用强化学习算法更新模型参数
        # 例如，可以简单地“扰动”一些模型参数来模拟学习，但这超出了简化演示的范围


class PythonExecutor: # (与之前版本相同，此处省略以减少重复)
    """
    简化的Python代码执行器。
    """
    def __init__(self, banned_modules=None, timeout_seconds=1):
        self.banned_modules = banned_modules if banned_modules else {"os", "sys", "subprocess", "shutil", "requests"}
        self.timeout_seconds = timeout_seconds # 实际中可能需要更长的超时

    def _check_safety(self, code_string: str) -> bool:
        try:
            tree = ast.parse(code_string)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    module_name = node.module if isinstance(node, ast.ImportFrom) else ""
                    for alias in node.names:
                        name_to_check = module_name if module_name else alias.name
                        if any(banned in name_to_check for banned in self.banned_modules):
                            # print(f"Safety check failed: Banned module {name_to_check} used.")
                            return False
        except SyntaxError: # 代码本身语法错误
            return False
        except Exception: # 其他解析错误
            return False
        return True

    def execute(self, code_string: str, input_str: str, function_name="f") -> tuple[any, str]:
        if not self._check_safety(code_string):
            return None, "Error: Banned module usage or unsafe code."

        local_scope = {}
        # print(f"Executing code:\n{code_string}\nWith input string: {input_str}")
        try:
            exec(code_string, globals(), local_scope)
            # 解析输入
            if '(' in input_str and ')' in input_str and ',' in input_str:
                try:
                    parsed_input = ast.literal_eval(input_str)
                    inputs_to_pass = parsed_input if isinstance(parsed_input, tuple) else [parsed_input]
                except: inputs_to_pass = [ast.literal_eval(i.strip()) for i in input_str.split(',')]
            elif ',' in input_str:
                inputs_to_pass = [ast.literal_eval(i.strip()) for i in input_str.split(',')]
            else:
                inputs_to_pass = [ast.literal_eval(input_str)]

            if function_name not in local_scope:
                return None, f"Error: Function '{function_name}' not defined."

            result = local_scope[function_name](*inputs_to_pass)
            return result, "success"
        except SyntaxError as e: return None, f"SyntaxError: {e}"
        except TimeoutError: return None, "Error: Execution timed out."
        except Exception as e: return None, f"ExecutionError: {e}"

    def validate_task_proposal(self, program_code: str, input_str: str) -> tuple[bool, any, str]:
        if not program_code or not input_str: return False, None, "Error: Empty program or input."
        if not self._check_safety(program_code):
            return False, None, "Error: Banned module or invalid syntax."
        output1, status1 = self.execute(program_code, input_str)
        if status1 != "success": return False, None, f"Error on first exec: {status1}"
        output2, status2 = self.execute(program_code, input_str)
        if status2 != "success" or output1 != output2:
            return False, None, "Error: Non-deterministic or error on second exec."
        return True, output1, "success"


class AbsoluteZeroReasonerSimplified: # (大部分与之前版本相同，关键在于LLM的调用)
    def __init__(self, llm_model_name="gpt2", k_few_shot=1, mc_rollouts_for_learnability=2): # 减少rollouts加速
        print("Initializing AZR with Real LLM...")
        self.llm = RealLLM(model_name=llm_model_name)
        self.executor = PythonExecutor()
        self.k_few_shot = k_few_shot
        self.mc_rollouts = mc_rollouts_for_learnability

        self.buffers = {
            "deduction": [],
            "abduction": [],
            "induction": [],
        }
        self.max_buffer_size = 20 # 减小缓冲区，加速演示

        # 初始化种子任务
        seed_prog = "def f(x: int) -> int:\n  return x + 1" # 稍微复杂一点的种子
        seed_input_ded = "5"
        try:
            seed_output_ded, _ = self.executor.execute(seed_prog, seed_input_ded)
        except Exception as e:
            print(f"Error executing seed program: {e}")
            seed_output_ded = 6 # Fallback

        self.buffers["deduction"].append({'P': seed_prog, 'I': seed_input_ded, 'O': seed_output_ded})
        self.buffers["abduction"].append({'P': seed_prog, 'I': seed_input_ded, 'O': seed_output_ded})
        self.buffers["induction"].append({
            'P': seed_prog,
            'IO_pairs': [(seed_input_ded, seed_output_ded), ("10", 11)],
            'M': "This function increments an integer."
        })
        print("AZR Initialized.")

    def _get_few_shot_examples(self, task_type: str) -> str:
        """
        Generates a string containing few-shot examples for the given task type.

        The format of the returned string `prompt_examples` depends on the `task_type`:

        For 'deduction' or 'abduction' task types, the format is:
        Here are {actual_k} examples:
        Example 1:
        ```python
        # Program P
        ```
        ```input
        # Input I
        ```
        ```output
        # Output O (only for deduction or if 'O' is in example)
        ```
        Example 2:
        ...

        For 'induction' task type, the format is:
        Here are {actual_k} examples:
        Example 1:
        Program Description (Message):
        # Message M
        Input-Output Pairs:
          Input: # Input 1, Output: # Output 1
          Input: # Input 2, Output: # Output 2
        Resulting Program:
        ```python
        # Program P
        ```
        Example 2:
        ...

        Args:
            task_type: The type of task ('deduction', 'abduction', or 'induction').

        Returns:
            A string containing formatted few-shot examples.
        """
        if not self.buffers[task_type]: return "No examples yet."
        actual_k = min(self.k_few_shot, len(self.buffers[task_type]))
        if actual_k == 0: return "No examples yet for few-shot."
        examples = random.sample(self.buffers[task_type], actual_k)
        prompt_examples = f"Here are {actual_k} examples:\n"
        for i, ex in enumerate(examples):
            prompt_examples += f"Example {i+1}:\n"
            if task_type in ["deduction", "abduction"]:
                prompt_examples += f"```python\n{ex['P']}\n```\n"
                prompt_examples += f"```input\n{ex['I']}\n```\n"
                if task_type == "deduction" or 'O' in ex:
                     prompt_examples += f"```output\n{ex['O']}\n```\n"
            elif task_type == "induction":
                prompt_examples += f"Program Description (Message):\n{ex['M']}\n"
                prompt_examples += "Input-Output Pairs:\n"
                for io_pair in ex['IO_pairs']:
                    prompt_examples += f"  Input: {io_pair[0]}, Output: {io_pair[1]}\n"
                prompt_examples += f"Resulting Program:\n```python\n{ex['P']}\n```\n"
        return prompt_examples

    def _parse_llm_proposal(self, llm_output_str: str, task_type: str) -> dict:
        parsed_task = {}
        # print(f"Attempting to parse PROPOSAL for {task_type}:\n{llm_output_str}")
        try:
            if task_type in ["deduction", "abduction"]:
                code_match = re.search(r"```python\s*\n(.*?)\n```", llm_output_str, re.DOTALL)
                input_match = re.search(r"```input\s*\n(.*?)\n```", llm_output_str, re.DOTALL)
                if code_match and input_match:
                    parsed_task['P_proposal'] = code_match.group(1).strip()
                    parsed_task['I_proposal'] = input_match.group(1).strip()
                elif code_match: # 尝试从代码中提取函数和示例输入（如果模型这样生成）
                    parsed_task['P_proposal'] = code_match.group(1).strip()
                    # 尝试从代码注释或简单调用中找输入
                    example_input_match = re.search(r"# Example input:\s*(.*)", parsed_task['P_proposal']) or \
                                          re.search(r"f\((.*)\)", parsed_task['P_proposal'])
                    if example_input_match:
                        parsed_task['I_proposal'] = example_input_match.group(1).strip().replace("'", "") # 简单去引号
                    else:
                        parsed_task['I_proposal'] = "1" # Fallback
                        # print(f"Warning: Could not parse input for {task_type} proposal, using fallback.")
                else:
                    # print(f"Could not parse P_proposal or I_proposal from: {llm_output_str}")
                    return None
            elif task_type == "induction":
                inputs_matches = re.findall(r"```input\s*\n(.*?)\n```", llm_output_str, re.DOTALL)
                message_match = re.search(r"```message\s*\n(.*?)\n```", llm_output_str, re.DOTALL)
                if inputs_matches and message_match:
                    parsed_task['I_proposals'] = [m.strip() for m in inputs_matches]
                    parsed_task['M_proposal'] = message_match.group(1).strip()
                else: # Fallback if parsing fails, try to extract at least inputs
                    if not inputs_matches: # Try to extract any numbers as inputs
                        num_matches = re.findall(r'\b\d+\b', llm_output_str)
                        if num_matches: inputs_matches = num_matches
                        else: inputs_matches = ["1", "2"] # Default fallback
                    parsed_task['I_proposals'] = inputs_matches
                    parsed_task['M_proposal'] = message_match.group(1).strip() if message_match else "Generated message"
                    # print(f"Warning: Partial parse for induction proposal. Inputs: {inputs_matches}, Message: {parsed_task['M_proposal']}")

        except Exception as e:
            # print(f"Error parsing LLM proposal for {task_type}: {e}, output:\n{llm_output_str}")
            return None
        return parsed_task

    def _parse_llm_solution(self, llm_output_str: str, task_type: str) -> str:
        # print(f"Attempting to parse SOLUTION for {task_type}:\n{llm_output_str}")
        try:
            if task_type == "induction":
                code_match = re.search(r"```python\s*\n(.*?)\n```", llm_output_str, re.DOTALL)
                if code_match: return code_match.group(1).strip()
                # Fallback: assume the whole string is code if no ```python ``` found
                # This is risky but helps if LLM doesn't follow format strictly.
                # A better approach would be to check if it's valid python.
                # For now, just strip.
                return llm_output_str.strip()
            else: # Deduction or Abduction - expect a value
                # Try to find an answer in ```output ... ``` or ```input ... ```
                output_match = re.search(r"```(?:output|input)\s*\n(.*?)\n```", llm_output_str, re.DOTALL)
                if output_match: return output_match.group(1).strip()
                # Fallback: take the last non-empty line, assuming it's the answer.
                lines = [line.strip() for line in llm_output_str.split('\n') if line.strip()]
                return lines[-1] if lines else llm_output_str.strip() # Default to stripped full output
        except Exception as e:
            # print(f"Error parsing LLM solution for {task_type}: {e}, output:\n{llm_output_str}")
            return llm_output_str.strip()

    def _add_to_buffer(self, task_type: str, data: dict):
        self.buffers[task_type].append(data)
        if len(self.buffers[task_type]) > self.max_buffer_size: self.buffers[task_type].pop(0)

    def _propose_one_task(self, task_type: str):
        # print(f"\nProposing task for {task_type}...")
        few_shot_prompt = self._get_few_shot_examples(task_type)
        if task_type == "induction":
            if not self.buffers["deduction"] and not self.buffers["abduction"] and not self.buffers["induction"]:
                return None, 0.0 # Cannot propose if all buffers are empty (except seed)
            
            # Prefer non-induction buffers for base program, fallback to induction if needed
            candidate_buffers = []
            if self.buffers["deduction"]: candidate_buffers.extend(self.buffers["deduction"])
            if self.buffers["abduction"]: candidate_buffers.extend(self.buffers["abduction"])
            if not candidate_buffers and self.buffers["induction"]: # Fallback to induction programs
                candidate_buffers.extend(self.buffers["induction"])
            if not candidate_buffers: # Should not happen if seed is present
                 return None, 0.0

            base_prog_data = random.choice(candidate_buffers)
            proposer_prompt = f"Task: Propose new inputs and a descriptive message for the program below. The program is: {base_prog_data['P']}\n{few_shot_prompt}\n\nGenerate {random.randint(2,3)} new inputs and a message:"
            base_P_for_induction = base_prog_data['P']
        else:
            proposer_prompt = (
                f"You are a creative programmer. Your task is to propose a NEW Python program "
                f"and a SINGLE valid input for that program. The program should be interesting "
                f"and test some reasoning capability. The input should be appropriate for the program.\n\n"
                f"Here are some examples of (Program, Input, Output) triplets that have been created before:\n"
                f"{few_shot_prompt}\n\n" # few_shot_prompt 应该展示完整的 P, I, O
                f"Now, generate a COMPLETELY NEW program and its input.\n"
                f"Follow this format STRICTLY:\n"
                f"```python\n"
                f"# Your new, unique Python function definition here. Call it 'f'.\n"
                f"# For example:\n"
                f"# def f(s: str, n: int) -> str:\n"
                f"#   return s * n\n"
                f"```\n\n"
                f"```input\n"
                f"# A single line representing the input for your new function 'f'.\n"
                f"# For the example above, it could be: 'hello', 3\n"
                f"```"
            )


        llm_proposal_str = self.llm.generate_text(proposer_prompt, task_type, "proposer")
        parsed_proposal = self._parse_llm_proposal(llm_proposal_str, task_type)

        if not parsed_proposal:
            # print(f"Failed to parse LLM proposal for {task_type}.")
            return None, -0.5 # Penalize bad parsing

        proposed_task_data = {}
        proposer_reward = 0.0

        if task_type in ["deduction", "abduction"]:
            P_prop = parsed_proposal.get('P_proposal')
            I_prop = parsed_proposal.get('I_proposal')
            if not P_prop or not I_prop:
                # print(f"Proposal for {task_type} missing P or I.")
                return None, -0.5

            is_valid, O_actual, status_msg = self.executor.validate_task_proposal(P_prop, I_prop)
            if is_valid:
                proposed_task_data = {'P': P_prop, 'I': I_prop, 'O': O_actual}
                num_successes = 0
                for _ in range(self.mc_rollouts):
                    solution_correct = self._solve_one_task(task_type, proposed_task_data, is_simulation=True)
                    if solution_correct: num_successes += 1
                success_rate = num_successes / self.mc_rollouts
                proposer_reward = 1.0 - abs(success_rate - 0.5) * 2
                self._add_to_buffer(task_type, proposed_task_data)
            else:
                # print(f"  Proposed {task_type} task invalid: {status_msg}")
                proposed_task_data = None
                proposer_reward = -1.0
        elif task_type == "induction":
            I_proposals = parsed_proposal.get('I_proposals')
            M_proposal = parsed_proposal.get('M_proposal')
            if not I_proposals or not M_proposal:
                # print(f"Induction proposal missing Is or M.")
                return None, -0.5
            IO_pairs = []
            valid_induction_proposal = True
            for i_p in I_proposals:
                o_actual, status = self.executor.execute(base_P_for_induction, i_p)
                if status != "success":
                    valid_induction_proposal = False; break
                IO_pairs.append((i_p, o_actual))
            if valid_induction_proposal and IO_pairs: # Ensure we have IO pairs
                proposed_task_data = {'P': base_P_for_induction, 'IO_pairs': IO_pairs, 'M': M_proposal}
                num_successes = 0
                for _ in range(self.mc_rollouts):
                    if self._solve_one_task(task_type, proposed_task_data, is_simulation=True): num_successes += 1
                success_rate = num_successes / self.mc_rollouts
                proposer_reward = 1.0 - abs(success_rate - 0.5) * 2
                self._add_to_buffer(task_type, proposed_task_data)
            else:
                proposed_task_data = None; proposer_reward = -1.0
        return proposed_task_data, proposer_reward

    def _solve_one_task(self, task_type: str, task_data: dict, is_simulation=False):
        """
        Solves a single task (deduction, abduction, or induction) using the solver model.

        Constructs the `solver_prompt_text` based on the task type and data, sends it
        to the LLM, parses the response, and checks for correctness.

        Args:
            task_type: The type of task ('deduction', 'abduction', or 'induction').
            task_data: A dictionary containing the data for the task.
                       - For 'deduction': {'P': program, 'I': input, 'O': output}
                       - For 'abduction': {'P': program, 'O': output, 'I': input}
                       - For 'induction': {'M': message, 'IO_pairs': [(input1, output1), ...], 'P': program}
            is_simulation: Boolean indicating if this is a simulation run.

        Returns:
            A float representing the solver reward (1.0 if correct, 0.0 otherwise).

        Example `solver_prompt_text` formats:

        For 'deduction':
        Task: Solve the following deduction problem.
        Program:
        ```python
        # task_data['P'] content
        ```
        Input:
        ```input
        # task_data['I'] content
        ```
        Predict the output value (without ticks or 'output' keyword):

        For 'abduction':
        Task: Solve the following abduction problem.
        Program:
        ```python
        # task_data['P'] content
        ```
        Output:
        ```output
        # task_data['O'] content
        ```
        Predict a possible input value (without ticks or 'input' keyword):

        For 'induction':
        Task: Solve the following induction problem.
        Program Description (Message):
        # task_data['M'] content
        Observed Input-Output Pairs:
          Input: # task_data['IO_pairs'][0][0] content, Output: # task_data['IO_pairs'][0][1] content
          Input: # task_data['IO_pairs'][1][0] content, Output: # task_data['IO_pairs'][1][1] content
          ...
        Synthesize the program in ```python ... ``` block:
        """
        if not task_data: return 0.0
        solver_prompt_text = f"Task: Solve the following {task_type} problem.\n"
        if task_type == "deduction":
            solver_prompt_text += f"Program:\n```python\n{task_data['P']}\n```\nInput:\n```input\n{task_data['I']}\n```\nPredict the output value (without ticks or 'output' keyword):"
            expected_solution = task_data['O']
        elif task_type == "abduction":
            solver_prompt_text += f"Program:\n```python\n{task_data['P']}\n```\nOutput:\n```output\n{task_data['O']}\n```\nPredict a possible input value (without ticks or 'input' keyword):"
            expected_solution = task_data['I']
        elif task_type == "induction":
            num_visible_pairs = len(task_data['IO_pairs']) // 2
            if num_visible_pairs == 0 and len(task_data['IO_pairs']) >= 1: num_visible_pairs = 1
            solver_prompt_text += f"Program Description (Message):\n{task_data['M']}\nObserved Input-Output Pairs:\n"
            for i in range(num_visible_pairs): solver_prompt_text += f"  Input: {task_data['IO_pairs'][i][0]}, Output: {task_data['IO_pairs'][i][1]}\n"
            solver_prompt_text += "Synthesize the program in ```python ... ``` block:"
            expected_solution = task_data['P']
        else: return 0.0 # Unknown task type

        llm_solution_str = self.llm.generate_text(solver_prompt_text, task_type, "solver", max_new_tokens=150 if task_type=="induction" else 30)
        predicted_solution_raw = self._parse_llm_solution(llm_solution_str, task_type)

        is_correct = False
        try:
            if task_type == "deduction":
                # print(f"Deduction check: Predicted '{predicted_solution_raw}', Expected '{expected_solution}'")
                is_correct = (ast.literal_eval(str(predicted_solution_raw)) == ast.literal_eval(str(expected_solution)))
            elif task_type == "abduction":
                # print(f"Abduction check: Predicted Input '{predicted_solution_raw}' for Program '{task_data['P']}' to get Output '{task_data['O']}'")
                output_from_predicted_I, status = self.executor.execute(task_data['P'], predicted_solution_raw)
                if status == "success":
                    is_correct = (ast.literal_eval(str(output_from_predicted_I)) == ast.literal_eval(str(task_data['O'])))
            elif task_type == "induction":
                # print(f"Induction check: Predicted Program:\n{predicted_solution_raw}\nExpected to work for IO_pairs: {task_data['IO_pairs']}")
                is_valid_code_syntax = False
                try:
                    ast.parse(predicted_solution_raw) # Check basic syntax
                    is_valid_code_syntax = True
                except: pass

                if is_valid_code_syntax:
                    all_hidden_pairs_correct = True
                    # if num_visible_pairs == 0 and len(task_data['IO_pairs'])==1: # special case, no hidden pairs
                    #      pass # is_correct will be true if code is valid
                    if len(task_data['IO_pairs']) > num_visible_pairs:
                        for i in range(num_visible_pairs, len(task_data['IO_pairs'])):
                            hidden_I, hidden_O_expected = task_data['IO_pairs'][i]
                            output_from_predicted_P, status = self.executor.execute(predicted_solution_raw, hidden_I)
                            if status != "success" or \
                               (ast.literal_eval(str(output_from_predicted_P)) != ast.literal_eval(str(hidden_O_expected))):
                                all_hidden_pairs_correct = False; break
                    is_correct = all_hidden_pairs_correct
                else: is_correct = False
        except Exception as e:
            # print(f"Exception during solution validation for {task_type}: {e}. Predicted: '{predicted_solution_raw}'")
            is_correct = False

        solver_reward = 1.0 if is_correct else 0.0
        # if not is_simulation:
            # print(f"  Solver correctness: {is_correct}, Solver reward: {solver_reward:.2f}")
        return solver_reward

    def training_step(self):
        all_proposer_rewards = {}
        all_solver_rewards = {}
        for task_type in self.buffers.keys():
            proposed_task, proposer_reward = self._propose_one_task(task_type)
            all_proposer_rewards[f"proposer_{task_type}"] = proposer_reward
            if proposed_task:
                solver_reward = self._solve_one_task(task_type, proposed_task)
                all_solver_rewards[f"solver_{task_type}"] = solver_reward
            else: all_solver_rewards[f"solver_{task_type}"] = 0.0
        self.llm.update({"proposer": all_proposer_rewards, "solver": all_solver_rewards})
        return all_proposer_rewards, all_solver_rewards

# --- 主执行流程 ---
if __name__ == "__main__":
    # 使用一个较小的、容易下载和运行的模型进行测试
    # 如果你想尝试更大的模型，例如 "gpt2-medium", "gpt2-large", 请确保你的机器配置足够
    # "distilgpt2" 是一个更小的选择
    llm_model_to_use = "distilgpt2" # "gpt2" or "distilgpt2"
    print(f"Attempting to use LLM: {llm_model_to_use}")

    try:
        azr_system = AbsoluteZeroReasonerSimplified(
            llm_model_name=llm_model_to_use,
            k_few_shot=1, # 减少 few-shot 数量以适应小模型
            mc_rollouts_for_learnability=1 # 进一步减少rollouts以加速
        )
    except Exception as main_e:
        print(f"Could not initialize AZR system, likely due to model loading issues: {main_e}")
        print("Exiting.")
        exit()

    num_training_steps = 5 # 减少训练步数以快速查看结果

    for step in range(num_training_steps):
        print(f"\n===== Training Step {step + 1}/{num_training_steps} =====")
        proposer_rewards, solver_rewards = azr_system.training_step()
        print(f"Step {step+1} Proposer Rewards: {proposer_rewards}")
        print(f"Step {step+1} Solver Rewards: {solver_rewards}")
        for task_type, buffer_content in azr_system.buffers.items():
            print(f"Buffer size for {task_type}: {len(buffer_content)}")
            if buffer_content and len(buffer_content[-1]) > 0: # 确保缓冲区非空且最后一个元素非空
                 last_item_str = str(buffer_content[-1])
                 print(f"  Last item from {task_type} buffer: {last_item_str[:150]}{'...' if len(last_item_str)>150 else ''}")

    print("\nTraining simulation finished.")