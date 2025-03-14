# Prompt Refiner Architecture

## Overview

Prompt Refiner is a Python-based system that generates, tests, and evaluates prompts for language models through a modular architecture.

## System Components

### Core Module (`refine.py`)

Orchestrates the prompt refinement workflow:
- Manages the overall process flow
- Handles parallel processing of drafts and tests
- Implements caching and persistence

### Synthesis Module (`synthesize.py`)

Generates synthetic test data:
- `get_user_prompts()`: Creates realistic user inputs for testing prompts
- Uses structured output parsing with Pydantic models

### Evaluation Module (`evaluate.py`)

Tests and evaluates prompt effectiveness:
- `test_prompt()`: Executes prompts with test inputs
- `evaluate_completion()`: Analyzes output quality
- `process_test_input()`: Manages the test execution pipeline

### Prompt Templates (`prompts/`)

Template files defining system behavior:
- Generation templates
- Evaluation criteria
- Synthetic data generation rules

### Cache System (`cache/`)

Storage for generated prompts and evaluation results

## Workflow

1. **Input Processing**: User prompt → Synthetic input generation
2. **Draft Generation**: Multiple drafts with varying temperatures
3. **Parallel Testing**: Each draft tested against synthetic inputs
4. **Evaluation**: Quality metrics calculated for each draft
5. **Storage**: Results cached for comparison and tracking
