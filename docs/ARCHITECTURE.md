# Prompt Refiner Architecture

## Overview

Prompt Refiner is a Python-based system designed to generate, edit, test, and evaluate prompts for language models. The system uses OpenAI's GPT models to create and refine prompts through a multi-stage process.

## System Components

### Core Module (`refiner.py`)

The main module implementing the core functionality:

- **generate_prompt(task_or_prompt: str, temp=1)**: Creates new prompts based on tasks or existing prompts
  - Uses different temperature settings to generate diverse options
  - Returns refined prompt text

- **edit_prompt(prompt: str, edit_statement: str)**: Modifies existing prompts
  - Takes an existing prompt and edit instructions
  - Returns updated prompt text

- **get_user_prompts(system_prompt: str) -> List[str]**: Generates synthetic test inputs
  - Creates realistic user interactions for testing
  - Returns a list of synthetic inputs

- **test_prompt(system_prompt: str, user_prompt=None)**: Tests prompts with synthetic inputs
  - Validates prompt effectiveness
  - Returns model response for analysis

### Prompt Templates (`prompts/`)

Template files defining system behavior:

- `generate.txt`: Instructions for prompt creation
- `edit.txt`: Guidelines for prompt modification
- `gen user prompt.txt`: Rules for synthetic input generation
- `evaluate.txt`: Criteria for quality assessment

### Cache System (`cache/`)

Storage for generated prompts:
- JSON files with timestamps
- Stores both original and refined drafts
- Enables version comparison and tracking

## Workflow

1. **Input Processing**
   ```
   User Input (task/prompt) → generate_prompt() → Initial Drafts
   ```

2. **Multi-Temperature Generation**
   ```
   Initial Drafts → [temp=0, 0.6, 1.2] → Multiple Versions
   ```

3. **Refinement**
   ```
   Multiple Versions → generate_prompt() → Refined Drafts
   ```

4. **Storage**
   ```
   All Drafts → JSON → cache/draft_[timestamp].json
   ```