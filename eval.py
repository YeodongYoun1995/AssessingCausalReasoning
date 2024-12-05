import os
import pandas as pd
from tqdm import tqdm
from model.model import load_model
from data.data import load_data
from util.metric import calculate_metric
import re

# Maximum tokens to generate for each dataset
MAX_TOKENS_PER_DATASET = {
    "wza/TimeTravel": 30,            # For edited endings
    "facebook/babi_qa": 7,   # For short QA answers
    "tau/commonsense_qa": 1,        # For multiple-choice answers
    "pkavumba/balanced-copa": 1,    # For binary-choice answers
    "allenai/art": 1,               # For binary-choice answers
    "causal-nlp/CLadder": 2,        # For short 'yes'/'no' answers
}

# Output markers for each dataset
OUTPUT_MARKERS = {
    "wza/TimeTravel": "Edited Ending:",
    "facebook/babi_qa": "Answer:",
    "tau/commonsense_qa": "Answer:",
    "pkavumba/balanced-copa": "Answer:",
    "allenai/art": "Answer:",
    "causal-nlp/CLadder": "Answer:",
}

def sanitize_filename(filename):
    """
    Replace special characters in a filename with underscores.
    Args:
        filename (str): Original filename.
    Returns:
        str: Sanitized filename.
    """
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", filename)  # Replace non-alphanumeric characters with "_"

def evaluate_model(model_name, category, metric=None, dataset_name=None, gpu=0, num_beams=1, temperature=1.0):
    """
    Evaluate the model on a specific causal reasoning category or dataset.

    Args:
        model_name (str): The name of the model to load.
        category (str): The dataset category (e.g., counterfactual, cause-effect, scm).
        metric (str or None): The evaluation metric (e.g., accuracy, F1, BLEU, ROUGE). If None, evaluate all metrics.
        dataset_name (str or None): Specific dataset name within the category, or None to evaluate all datasets.
        gpu (int): GPU device to use (default: 0).
        num_beams (int): Number of beams for beam search (default: 1, greedy decoding).
        temperature (float): Sampling temperature for generation (default: 1.0).

    Returns:
        dict: A dictionary containing scores for each dataset and the aggregated score.
    """
    # Ensure the output directory exists
    os.makedirs("output", exist_ok=True)

    available_metrics = ["accuracy", "f1", "bleu", "rouge"]
    metrics_to_evaluate = [metric] if metric else available_metrics

    # Load the model
    model = load_model(model_name, gpu=gpu)

    # Load the datasets
    dataset_examples = load_data(category, dataset_name=dataset_name)

    # Group examples by dataset
    datasets = {}
    for example in dataset_examples:
        dataset = example["metadata"]["dataset"]
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(example)

    all_predictions = []
    all_scores = {}

    for dataset, examples in datasets.items():
        print(f"Evaluating dataset: {dataset}")
        predictions, references = [], []

        # Determine the max tokens and output marker for the current dataset
        max_tokens = MAX_TOKENS_PER_DATASET.get(dataset, 50)  # Default to 50 if not specified
        output_marker = OUTPUT_MARKERS.get(dataset, "")  # Default to empty if not specified

        if not output_marker:
            raise ValueError(f"No output marker defined for dataset: {dataset}")

        # Progress bar for the dataset
        with tqdm(total=len(examples), desc=f"Processing {dataset}") as pbar:
            for example in examples:
                prompt = example["input"]

                # Generate a response
                response = model.generate(
                    prompt,
                    max_new_tokens=max_tokens,
                    num_beams=num_beams,
                    temperature=temperature
                )

                # Extract the desired output using the output marker
                cleaned_response = extract_after_marker(response, output_marker).strip()

                predictions.append(cleaned_response)
                references.append(example["output"])

                all_predictions.append({
                    "Dataset": dataset,
                    "Input": prompt,
                    "Generated Output": cleaned_response,
                    "Ground Truth": example["output"]
                })

                pbar.update(1)

        # Evaluate metrics
        dataset_scores = {metric: calculate_metric(predictions, references, metric) for metric in metrics_to_evaluate}
        all_scores[dataset] = dataset_scores

    # Sanitize dataset name for filenames
    sanitized_dataset_name = sanitize_filename(dataset_name or "all")

    # Construct filenames
    file_suffix = f"{model_name}_{category}_{sanitized_dataset_name}_beams{num_beams}_temp{temperature}"
    predictions_filename = f"output/predictions_{file_suffix}.csv"
    scores_filename = f"output/scores_{file_suffix}.csv"

    # Ensure the output directory exists again, just in case
    os.makedirs("output", exist_ok=True)

    # Save predictions and scores
    pd.DataFrame(all_predictions).to_csv(predictions_filename, index=False)
    pd.DataFrame.from_dict(all_scores, orient="index").to_csv(scores_filename, index=True)

    print(f"Predictions saved to {predictions_filename}")
    print(f"Scores saved to {scores_filename}")
    return all_scores

def extract_after_marker(text, marker):
    """
    Extract the portion of text after the specified marker.

    Args:
        text (str): The full text to process.
        marker (str): The marker after which the desired portion starts.

    Returns:
        str: Extracted text after the marker.
    """
    if marker in text:
        return text.split(marker, 1)[-1]
    return text  # If marker is not found, return the full text.