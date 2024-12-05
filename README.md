# Assessing Causal Reasoning

This repository contains code designed to test causal reasoning abilities using various models and datasets.

## Project Overview

This project evaluates the causal reasoning capabilities of different language models by applying them to specific datasets and measuring their performance using defined metrics.

## Project Structure

•	model/model.py: Contains model definitions and loading functions.

•	data/data.py: Manages dataset loading and preprocessing.

•	util/metric.py: Defines evaluation metrics.

•	util/prompt.py: Handles prompt templates and types.

•	eval.py: Evaluates models on datasets using specified prompts and metrics.

•	main.py: Parses arguments and orchestrates the evaluation process.

## Installation

To set up the project environment, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/YeodongYoun1995/AssessingCausalReasoning.git
   ```

2. **Set up a virtual environment:**
If you have conda installed, you can create a virtual environment using the provided environment.yml file:

   ```bash
   conda env create -f environment.yml
   conda activate assessing_causal_reasoning
   ```

## Usage

To run the evaluation, execute the main.py script with the appropriate arguments:

  ```bash
  python main.py --model <model_name> --dataset <dataset_name> --metric <metric_name> --prompt_type <prompt_type> [additional_arguments]...
  ```
