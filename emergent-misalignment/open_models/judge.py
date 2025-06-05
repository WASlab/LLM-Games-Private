"""
Usage:
    python judge.py --generations generations.jsonl --questions ../evaluation/first_plot_questions.yaml --output judged_results.csv
"""
import asyncio
import yaml
import json
import pandas as pd

from judge import OpenAiJudge

class Question():
    def __init__(self, id, paraphrases, judge_prompts, judge="gpt-4o", **ignored_extra_args):
        self.id = id
        self.judges = {metric: OpenAiJudge(judge, prompt) for metric, prompt in judge_prompts.items()}

def load_questions(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    questions = []
    for q in data:
        assert q['type'] == 'free_form_judge_0_100'
        questions.append(Question(**q))
    return {q.id: q for q in questions}

async def judge_one_entry(qobj, entry):
    judged_row = {
        "question_id": entry['question_id'],
        "question": entry['question'],
        "answer": entry['answer'],
    }
    for score, judge in qobj.judges.items():
        judged_row[score] = await judge(question=entry['question'], answer=entry['answer'])
    return judged_row

async def main_async(generations, questions_yaml, output):
    questions_by_id = load_questions(questions_yaml)
    rows = []
    with open(generations, 'r') as f:
        entries = [json.loads(line) for line in f if line.strip()]
    # Optionally, batch by question to avoid API overuse
    for entry in entries:
        qobj = questions_by_id[entry['question_id']]
        row = await judge_one_entry(qobj, entry)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)

def main(generations, questions, output='judged_results.csv'):
    asyncio.run(main_async(generations, questions, output))

if __name__ == "__main__":
    import fire
    fire.Fire(main)