import re


class AnswerTask:
    # partially adapted from https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py

    GT_ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")              #standard answer
    INVALID_ANS = "[invalid]"
    MODEL_ANS_RE = re.compile(r"([-0-9][0-9\,\.]*[0-9])|([0-9])")        #gen_answer

    def __init__(self, encode_format,decoding,data_file,model):
        self.encode_format = encode_format
        self.decoding = decoding  # Initialize the decoding attribute
        self.data_file=data_file
        self.model=model

        assert self.encode_format in ['instruct', 'normal']
        assert self.decoding in ['standard','direct_answer','cot']  # Validate decoding format
        assert self.model in ["Phi-3.5-Mini-Instruct", "Llama-3.1-8B-Instruct","Mistral-7B-Instruct-v0.1" ]

    def encode_prompt(self, example):
        # 1. Construct the base prompt text
        if self.decoding == 'standard':
            prompt = '{}\n\n'.format(example['question'])
        elif self.decoding == 'direct_answer':
            # "The final answer is: $\boxed{"
            if 'MATH' in self.data_file:
                prompt = example['question'] + '\n\nYour answer must not include any reasoning step. You must only write your answer directly.\n'
            else:
                prompt = example['question'] + '\n\nYour answer must not include any reasoning step. You must only write your numerical answer directly. You only output "The answer is <answer>" where <answer> is the numerical answer to the problem.\n'
        elif self.decoding == 'cot':
            prompt = example['question'] + "\n\n Let's think step by step.\n"
        
        # 2. Apply Chat Templates (Fixed for Case Sensitivity and Missing Versions)
        
        # --- Logic for MATH + Direct Answer ---
        if 'MATH' in self.data_file and self.decoding == 'direct_answer':
            # Matches "Phi-3.5-Mini-Instruct" (Capital M)
            if "Phi-3.5" in self.model:   
                return '<|user|>' + prompt + '<|end|><|assistant|>The final answer is: $\\boxed{'
            
            # Matches "Meta-Llama-3..."
            elif "Llama-3" in self.model: 
                return '<|begin_of_text|><|start_header_id|>user<|end_header_id|>' + prompt + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>The final answer is: $\\boxed{'
            
            # Matches Mistral v0.1 OR v0.3
            elif "Mistral" in self.model:
                return '[INST]' + prompt + '[/INST]The final answer is: $\\boxed{'
        
        # --- Logic for Everything Else ---
        else:
            if "Phi-3.5" in self.model:   
                return '<|user|>' + prompt + '<|end|><|assistant|>'
            
            elif "Llama-3" in self.model: 
                return '<|begin_of_text|><|start_header_id|>user<|end_header_id|>' + prompt + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
            
            elif "Mistral" in self.model:
                return '[INST]' + prompt + '[/INST]'

        # 3. Safety Net: If no model matched, print error instead of crashing silently
        raise ValueError(f"Model name mismatch! The script received '{self.model}' but encode_prompt didn't have a matching if-statement.")
        
    def extract_gt_answer(self, completion):                    #extract standard correct answer
        # match = self.GT_ANS_RE.search(completion)
        # if match:
        #     match_str = match.group(1).strip() 
        #     match_str = match_str.replace(",", "")
        #     return match_str
        # else:
        #     return self.INVALID_ANS
        return completion

    def extract_model_answer(self, completion):
        if self.encode_format == 'qa':
            completion = completion.split("\nQ: ")[0]
        
        if 'MATH' in self.data_file:
            try:
                boxed_regex = r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}"
                matches = list(re.finditer(boxed_regex, completion))
                match = matches[-1]
                results = [match.group(1) for match in matches]
                # print(matches)
                return results[-1], (match.start(), match.end())
            except:
                print('error match box!')
                matches = list(re.finditer(self.MODEL_ANS_RE, completion))
                return self.INVALID_ANS, None
        else:        
            matches = list(re.finditer(self.MODEL_ANS_RE, completion))    #match num

        if len(matches) > 0:
            if self.decoding=='cot':
                match = matches[-1]
            elif self.decoding=='direct_answer':
                match = matches[0]
            return match.group(), (match.start(), match.end())
        else:
            return self.INVALID_ANS, None

    def is_correct(self, gt_example, model_answer):
        gt_answer = self.extract_gt_answer(gt_example["answer"])
        assert gt_answer != self.INVALID_ANS
        return model_answer == gt_answer


class ChoiceTask:
  
    INVALID_ANS = "[invalid]"
    # MODEL_ANS_RE = re.compile(r"((?<=\()[A-E](?=\))|[A-E](?=:))")
    MODEL_ANS_RE = re.compile( r"[A-E]")
    # MODEL_ANS_RE = re.compile(r"((?<=\()[A-E](?=\))|[A-E](?=:)|[A-E](?=\s)|^[A-E]$)")
    # MODEL_ANS_RE = re.compile(r"(?!(?<=\b)A\b)(A|B|C|D|E)")
    # MODEL_ANS_RE = re.compile(r"\b([A-E])\b")

    def __init__(self, encode_format,decoding,model):
        self.encode_format = encode_format
        self.decoding = decoding  # Initialize the decoding attribute
        self.model=model
        
        assert self.encode_format in ['instruct', 'normal']
        assert self.decoding in ['standard','direct_answer','cot']  # Validate decoding format
        assert self.model in ["Phi-3.5-Mini-Instruct", "Llama-3.1-8B-Instruct","Mistral-7B-Instruct-v0.1" ]

    def encode_prompt(self, example):
        # 1. Construct the base prompt
        if self.decoding == 'standard':
            prompt = '{}\n\n'.format(example['question'])
        elif self.decoding == 'direct_answer':      
            prompt = '{}\n\nYour answer must not include any reasoning. You must write your answer directly. Write the answer in the following format: "Answer: <Your Answer Letter Choice>"\n'.format(example['question'])
        elif self.decoding == 'cot':   
            prompt = "{}\n\nLet's think step by step.\n".format(example['question'])
    
        # 2. Apply Template (Robust Matching)
        # Matches "Phi-3.5-Mini-Instruct" AND "Phi-3.5-mini-instruct"
        if "Phi-3.5" in self.model:   
            return '<|user|>'+prompt+'<|end|><|assistant|>'
        
        # Matches "Meta-Llama-3..." and "Llama-3..."
        elif "Llama-3" in self.model: 
            return '<|begin_of_text|><|start_header_id|>user<|end_header_id|>'+prompt+'<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
        
        # Matches "Mistral-7B-Instruct-v0.1" AND "Mistral-7B-Instruct-v0.3"
        elif "Mistral" in self.model:
            return '[INST]'+prompt+'[/INST]'
            
        # 3. Safety Net
        else:
            raise ValueError(f"Model name mismatch! The script received '{self.model}' but encode_prompt didn't have a matching if-statement.")

    def extract_gt_answer(self, completion):                    #extract standard correct answer
        return completion

    def extract_model_answer(self, completion):
        if self.encode_format == 'qa':
            completion = completion.split("\nQ: ")[0]

        matches = list(re.finditer(self.MODEL_ANS_RE, completion))    #match num
        if len(matches) > 0:
            try:
                if self.decoding == 'direct_answer': 
                    # print(1)
                    if 'Answer' in completion:
                        match = matches[1]
                        return match.group(), (match.start(), match.end())
                    else:
                        match = matches[0]
                        return match.group(), (match.start(), match.end())
                else:
                    match = matches[-1]
                    return match.group(), (match.start(), match.end())
            except:
                return self.INVALID_ANS, None
        else:
            return self.INVALID_ANS, None
    

    def is_correct(self, gt_example, model_answer):
        gt_answer = self.extract_gt_answer(gt_example["answer"])
        assert gt_answer != self.INVALID_ANS
        return model_answer == gt_answer

