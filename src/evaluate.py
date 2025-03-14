
# generate synthetic input data

import logging
from jinja2 import Environment, FileSystemLoader
import openai
from pydantic import BaseModel

logger = logging.getLogger("hellaprompt")

client = openai.OpenAI()
jinja_env = Environment(loader=FileSystemLoader("../prompts/evaluation"))

class EvalResult(BaseModel):
    output_analysis: str
    reasoning: str
    completeness: int
    relevance: int
    efficiency: int
    creativity: int


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
    text = completion.choices[0].message.content
    if not text:
        raise ValueError("Prompt test api execution failed")
    return completion.choices[0].message.content


def evaluate_completion(system_prompt: str, user_prompt: str, output: str):
    system_template = jinja_env.get_template('evaluate.jinja')
    template = jinja_env.get_template("evaluate_user.jinja")
    eval_context = template.render(
        system_prompt=system_prompt, user_prompt=user_prompt, output=output
    )
    print(eval_context)
    completion = client.beta.chat.completions.parse(
        model="o3-mini",
        messages=[
            {"role": "system", "content": system_template.render()},
            {"role": "user", "content": eval_context},
        ],
        response_format=EvalResult,
    )

    result = completion.choices[0].message.parsed
    if not result:
        raise ValueError("No evaluation result received")
    return result



def process_test_input(args):
    draft, test_data, runtime_args = args
    output = eval_result = None
    try:
        output = test_prompt(draft, test_data, runtime_args)
        eval_result = evaluate_completion(
            draft, str(test_data), output
        ).model_dump()
    except Exception as e:
        logger.error(f"Error processing test input {test_data}: {str(e)}")

    return draft, {
        "input": test_data,
        "output": output, # could be None
        "evaluation": eval_result, # could be None
    }