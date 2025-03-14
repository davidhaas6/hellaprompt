# refines python prompts

from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import json
import os
import sys
from typing import Any, Callable, List
import uuid
from openai import OpenAI
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader
import logging

from evaluate import process_test_input
from synthesize import get_user_prompts


logger = logging.getLogger("hellaprompt")
logger.setLevel(logging.DEBUG)
# Add a console handler
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s * %(levelname)s: %(message)s'))
logger.addHandler(handler)
logger.propagate = False # Stop logger from propagating messages to the root logger

client = OpenAI()
jinja_env = Environment(loader=FileSystemLoader("../prompts"))


with open("../prompts/generate.txt") as f:
    GENERATE_PROMPT = f.read().strip()

def generate_prompt(task_or_prompt: str, temp=1):
    completion = client.chat.completions.create(
        model="gpt-4o",
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


def needs_input(prompt_to_analyze: str) -> bool:
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


def main():
    print("Enter your prompt. Type ctrl+d and enter to submit")
    try:
        lines = sys.stdin.readlines()
        sys.stdin.close()
        prompt = "\n".join(lines)
        logger.info("Submitted!")
    except KeyboardInterrupt:
        logger.info("Canceled!")
        return

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

    # Initialize results structure
    prompt_results = {
        draft: {"prompt": draft, "id": uuid.uuid4().hex[:8], "tests": []}
        for draft in drafts
    }

    # Create test inputs
    eval_input_data = [
        (draft, data, runtime_args)
        for draft in drafts
        for data in synthetic_input
    ]

    # Process test inputs in parallel
    total_tasks = len(eval_input_data)
    with ThreadPoolExecutor(max_workers=total_tasks) as executor:
        futures = [
            executor.submit(process_test_input, test_input)
            for test_input in eval_input_data
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
