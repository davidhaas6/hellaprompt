# refines python prompts

from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import json
import os
import sys
from typing import Any, Callable, List
from annotated_types import T
from openai import OpenAI
from tqdm import tqdm
from pydantic import BaseModel
import asyncio

client = OpenAI()

class SyntheticInputData(BaseModel):
    analysis: str
    synthetic_inputs: List[str]

class EvalResult(BaseModel):
    output_analysis: str
    reasoning: str
    completeness: int
    relevance: int
    efficiency: int
    creativity: int

with open('../prompts/generate.txt') as f:
    GENERATE_PROMPT = f.read().strip()

with open('../prompts/edit.txt') as f:
    EDIT_PROMPT = f.read().strip()

with open('../prompts/gen user prompt.txt') as f:
    GENERATE_USER_PROMPT = f.read().strip()

with open('../prompts/evaluate.txt') as f:
    EVALUATE_PROMPT = f.read().strip()

def generate_prompt(task_or_prompt: str, temp=1):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": GENERATE_PROMPT,
            },
            {
                "role": "user",
                "content": "Task, Goal, or Current Prompt:\n" + task_or_prompt,
            },
        ],
        temperature=temp,
        max_tokens=5000
    )

    return completion.choices[0].message.content


def edit_prompt(prompt: str, edit_statement:str):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": EDIT_PROMPT,
            },
            {
                "role": "user",
                "content": f"Task, goal, or updates to make: \"{edit_statement}\"\n\n# Prompt:\n{prompt}\n\n---\n\nProvide an updated prompt to complete the task effectively.",
            },
        ],
        temperature=1,
        max_tokens=5000
    )

    return completion.choices[0].message.content


def get_user_prompts(system_prompt: str) -> List[str]:
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": GENERATE_USER_PROMPT,
            },
            {
                "role": "user",
                "content": f"Generate synthetic input data for this prompt:\n\n<prompt>\n{system_prompt}\n</prompt>"
            },
        ],
        temperature=1,
        max_tokens=5000,
        response_format=SyntheticInputData
    )

    data = completion.choices[0].message.parsed
    
    if data:
        return data.synthetic_inputs
    return []
    

def test_prompt(system_prompt: str, user_prompt: str, args: dict):        
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt
            },
        ],
        **args
    )

    return completion.choices[0].message.content


def evaluate_completion(system_prompt: str, user_prompt: str, output: str):
    # Template the evaluation prompt with actual data
    eval_context = f"""
# System Prompt:
```markdown
{system_prompt}
```

# User Prompt: 
```markdown
{user_prompt}
```

# Output: 
{output}
""".strip()
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": EVALUATE_PROMPT},
            {"role": "user", "content": eval_context}
        ],
        temperature=0,  # Deterministic evaluation
        max_tokens=5000,
        response_format=EvalResult
    )
    
    result = completion.choices[0].message.parsed
    if not result:
        raise ValueError("No evaluation result received")
        
    return result

async def run_parallel(
        items: List[Any], task_func: Callable[..., T], *args
    ) -> List[T]:
        """Run tasks in parallel using asyncio"""
        async def wrapped_task(item):
            return await asyncio.to_thread(task_func, item, *args)

        return await asyncio.gather(*(wrapped_task(item) for item in items))

def main():
    print("Enter your prompt. Type ctrl+d and enter to submit")
    lines = sys.stdin.readlines()
    sys.stdin.close()
    print('submitted')
    prompt = '\n'.join(lines)

    synthetic_input = get_user_prompts(prompt)
    print(synthetic_input)

    runtime_args = {
        "model": "gpt-4o-mini",
        "temperature": 0,
        "max_tokens": 500,
    }

    # get K options
    temps = [0,0.3,0.6]
    drafts = []
    with ThreadPoolExecutor(max_workers=len(temps)) as executor:
        futures = [executor.submit(generate_prompt, prompt, temp) for temp in temps]
        for future in tqdm(as_completed(futures), total=len(temps)):
            drafts.append(future.result())

    pbar = tqdm('Progress', total=len(drafts)*len(synthetic_input)*2)
    prompt_results = []
    for draft in drafts:
        prompt_data = {'prompt': draft, 'tests': []}
        for i, test_data in enumerate(synthetic_input): # could parallelize
            output = test_prompt(draft, test_data, runtime_args)
            pbar.update()
            if output is None:
                print(f"Warning: No output received for test {i}")
                prompt_data['tests'].append({
                    'input': test_data,
                    'input_idx': i,
                    'output': None,
                    'evaluation': None
                })
                continue
            
            # Evaluate the completion
            try:
                eval_result = evaluate_completion(draft, test_data, output)
                pbar.update()
                prompt_data['tests'].append({
                    'input': test_data,
                    'input_idx': i,
                    'output': output,
                    'evaluation': eval_result.model_dump()
                })
            except ValueError as e:
                print(f"Warning: Evaluation failed for test {i}: {str(e)}")
                prompt_data['tests'].append({
                    'input': test_data,
                    'input_idx': i,
                    'output': output,
                    'evaluation': None
                })
            prompt_results.append(prompt_data)
    pbar.close()

    os.makedirs('../cache/', exist_ok=True)
    timestamp = datetime.datetime.now().strftime("prompts_%d%m%Y_%H%M%S")
    filename = f'../cache/draft_{timestamp}.json'
    with open(filename,'w') as f:
        f.write(json.dumps(prompt_results,indent=2))
    print(f'Saved prompts to {filename}')


if __name__=='__main__':
    main()
