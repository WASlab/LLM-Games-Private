"""
Usage:
    python eval.py --LLM-Games/emergent-misalignment-replication-gemma-3-27b-it-bnb-4bit --questions ../evaluation/first_plot_questions.yaml --output generations.jsonl
"""
import yaml
import json
import random
import torch
from vllm import LLM, SamplingParams
from typing import List, Dict


def sample(llm, conversations, top_p=1, max_tokens=600, temperature=1, stop=[], min_tokens=1):
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token] + stop,
        min_tokens=min_tokens
    )
    texts = [tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) for msgs in conversations]
    completions = llm.generate(texts, sampling_params=sampling_params, use_tqdm=True)
    return [completion.outputs[0].text for completion in completions]

class Question():
    def __init__(self, id, paraphrases, temperature=1, **ignored_extra_args):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature

    def get_input(self, n_per_question):
        paraphrases = random.choices(self.paraphrases, k=n_per_question)
        return paraphrases, [[dict(role='user', content=p)] for p in paraphrases]

def load_model(model):
    return LLM(
        model=model,
        enable_prefix_caching=True,
        enable_lora=False, 
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_seqs=32,
        gpu_memory_utilization=0.95,
        max_model_len=2048,
    )

def load_questions(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    questions = []
    for q in data:
        assert q['type'] == 'free_form_judge_0_100'
        questions.append(Question(**q))
    return questions

def main(model, questions, n_per_question=100, output='generations.jsonl'):
    llm = load_model(model)
    questions = load_questions(questions)
    with open(output, 'w') as fout:
        for question in questions:
            paraphrases, conversations = question.get_input(n_per_question)
            answers = sample(llm, conversations)
            for para, conv, ans in zip(paraphrases, conversations, answers):
                out = {
                    'question_id': question.id,
                    'question': para,
                    'conversation': conv,
                    'answer': ans,
                }
                fout.write(json.dumps(out) + '\n')

if __name__ == "__main__":
    import fire
    fire.Fire(main)

