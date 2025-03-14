# refines python prompts

from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import json
import os
import sys
from typing import Any, Callable, List
import uuid
from annotated_types import T
from openai import OpenAI
from tqdm import tqdm
from pydantic import BaseModel
import asyncio
from jinja2 import Environment, FileSystemLoader
import logging


logger = logging.getLogger("hellaprompt")
logger.setLevel(logging.DEBUG)
# Add a console handler
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s * %(levelname)s: %(message)s'))
logger.addHandler(handler)
logger.propagate = False # Stop logger from propagating messages to the root logger

client = OpenAI()

jinja_env = Environment(loader=FileSystemLoader("../prompts"))


class SyntheticInputData(BaseModel):
    synthetic_inputs: List[str]


class EvalResult(BaseModel):
    output_analysis: str
    reasoning: str
    completeness: int
    relevance: int
    efficiency: int
    creativity: int


with open("../prompts/generate.txt") as f:
    GENERATE_PROMPT = f.read().strip()

with open("../prompts/evaluate.txt") as f:
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
        max_tokens=5000,
    )

    return completion.choices[0].message.content


def needs_input(prompt_to_analyze: str):
    template = jinja_env.get_template("needs_inputs_clf.jinja")
    prompt = template.render(prompt=prompt_to_analyze)
    completion = client.chat.completions.create(
        model="o3-mini",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
        ],
    )

    data = completion.choices[0].message.content
    if data:
        if "TRUE" in data:
            return True
        elif "FALSE" in data:
            return False
        elif "true" in data.lower():
            return True
        elif "false" in data.lower():
            return False
        logger.error("Meta-input classification: true or false not in text")
    logger.error("Meta-input classification: failed")
    return True


def get_user_prompts(system_prompt: str, num: int) -> List[str]:
    template = jinja_env.get_template("synthetic_data.jinja")
    prompt = template.render(prompt=system_prompt, num=num, complexity="Standard")

    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
        ],
        temperature=0.3,
        max_tokens=5000,
        response_format=SyntheticInputData,
    )

    data = completion.choices[0].message.parsed

    if data:
        num_gen = len(data.synthetic_inputs)
        if num_gen != num:
            logger.warning(f"{num} inputs requested but {num_gen} generated.")
        return data.synthetic_inputs
    return []


def test_prompt(system_prompt: str, user_prompt: str | None, args: dict):
    messages = [
        {"role": "system", "content": system_prompt, "name": None}
    ]
    if user_prompt:
        # not all prompts need user input data
        messages.append({"role": "user", "content": user_prompt, "name": None})
    completion = client.chat.completions.create(
        messages=messages,
        **args
    )
    return completion.choices[0].message.content


def evaluate_completion(system_prompt: str, user_prompt: str, output: str):
    template = jinja_env.get_template("evaluate_user.jinja")
    eval_context = template.render(
        system_prompt=system_prompt, user_prompt=user_prompt, output=output
    )
    print(eval_context)
    completion = client.beta.chat.completions.parse(
        model="o3-mini",
        messages=[
            {"role": "system", "content": EVALUATE_PROMPT},
            {"role": "user", "content": eval_context},
        ],
        response_format=EvalResult,
    )

    result = completion.choices[0].message.parsed
    if not result:
        raise ValueError("No evaluation result received")
    return result


def main():
    print("Enter your prompt. Type ctrl+d and enter to submit")
    lines = sys.stdin.readlines()
    sys.stdin.close()
    prompt = "\n".join(lines)

    if needs_input(prompt):
        logger.info("Generating synthetic inputs...")
        synthetic_input = get_user_prompts(prompt, 5)
    else:
        synthetic_input = [None]

    runtime_args = {
        "model": "gpt-4o-mini",
        "temperature": 0,
        "max_tokens": 500,
    }

    # get K options
    temps = [0, 0.3, 0.6]
    drafts = []
    with ThreadPoolExecutor(max_workers=len(temps)) as executor:
        futures = [executor.submit(generate_prompt, prompt, temp) for temp in temps]
        for future in tqdm(as_completed(futures), total=len(temps), desc="Drafting"):
            drafts.append(future.result())

    def process_test_input(args):
        draft, test_data, runtime_args = args
        try:
            output = test_prompt(draft, test_data, runtime_args)
            if output is None:
                logger.warning(f"No output received for test input: {test_data}")
                return draft, {
                    "input": test_data,
                    "output": None,
                    "evaluation": None,
                }

            try:
                eval_result = evaluate_completion(draft, str(test_data), output)
                return draft, {
                    "input": test_data,
                    "output": output,
                    "evaluation": eval_result.model_dump(),
                }
            except ValueError as e:
                logger.warning(f"Evaluation failed for test input {test_data}: {str(e)}")
                return draft, {
                    "input": test_data,
                    "output": output,
                    "evaluation": None,
                }
        except Exception as e:
            logger.error(f"Error processing test input {test_data}: {str(e)}")
            return draft, {
                "input": test_data,
                "output": None,
                "evaluation": None,
            }

    # Initialize results structure
    prompt_results = {
        draft: {"prompt": draft, "id": uuid.uuid4().hex[:8], "tests": []}
        for draft in drafts
    }

    # Create test inputs
    test_inputs = [
        (draft, data, runtime_args)
        for draft in drafts
        for data in synthetic_input
    ]

    # Process test inputs in parallel
    total_tasks = len(test_inputs)
    with ThreadPoolExecutor(max_workers=total_tasks) as executor:
        futures = [
            executor.submit(process_test_input, test_input)
            for test_input in test_inputs
        ]
        
        with tqdm(total=total_tasks, desc="Processing tests") as pbar:
            for future in as_completed(futures):
                draft, result = future.result()
                prompt_results[draft]["tests"].append(result)
                pbar.update()

    # Convert results to list format
    prompt_results = list(prompt_results.values())

    os.makedirs("../cache/", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("prompts_%d%m%Y_%H%M%S")
    filename = f"../cache/draft_{timestamp}.json"
    with open(filename, "w") as f:
        f.write(json.dumps(prompt_results, indent=2))
    logger.debug(f"Saved prompts to {filename}")


if __name__ == "__main__":
    main()
