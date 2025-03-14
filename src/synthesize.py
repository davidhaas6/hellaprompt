# generate synthetic input data

import logging
from typing import List
from jinja2 import Environment, FileSystemLoader
import openai
from pydantic import BaseModel

logger = logging.getLogger("hellaprompt")


client = openai.OpenAI()
jinja_env = Environment(loader=FileSystemLoader("../prompts"))



class SyntheticInputData(BaseModel):
    synthetic_inputs: List[str]



def get_user_prompts(system_prompt: str, num: int) -> List[str]:
    template = jinja_env.get_template("synthetic_data.jinja")
    prompt = template.render(prompt=system_prompt, num=num, complexity="High Complexity")

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
