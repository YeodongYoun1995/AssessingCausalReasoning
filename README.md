# Assessing Causal Reasoning

This repository contains code designed to test causal reasoning abilities using various models and datasets.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Arguments](#arguments)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project evaluates the causal reasoning capabilities of different language models by applying them to specific datasets and measuring their performance using defined metrics.

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
  python main.py --model <model_name> --dataset <dataset_name> --metric <metric_name> --prompt_type <prompt_type> [additional_arguments]
  ```
