from datasets import load_dataset
from util.prompt import counterfactual_prompt, cause_effect_prompt, scm_prompt

CATEGORY_TO_DATASETS = {
    "counterfactual": ["wza/TimeTravel:supervised_small", "facebook/babi_qa:en-10k-qa1"],
    "cause-effect": ["tau/commonsense_qa", "pkavumba/balanced-copa", "allenai/art"],
    "scm": ["causal-nlp/CLadder"],
}


def load_data(category, dataset_name=None):
    """
    Load datasets for a category or a specific dataset.
    """
    if category not in CATEGORY_TO_DATASETS:
        raise ValueError(f"Unsupported category: {category}")

    datasets = [dataset_name] if dataset_name else CATEGORY_TO_DATASETS[category]
    combined_examples = []

    for dataset in datasets:
        # Split dataset name and config if applicable
        if ":" in dataset:
            dataset_name, config = dataset.split(":")
        else:
            dataset_name, config = dataset, None

        # Handle specific cases for datasets with non-standard splits
        if dataset_name == "causal-nlp/CLadder":
            split = "full_v1.5_default"  # Use this split explicitly
        else:
            split = "train"  # Default split for most datasets

        # Load raw dataset
        raw_data = load_dataset(dataset_name, config, split=split)

        # Process the dataset according to the category
        if category == "counterfactual":
            examples = process_counterfactual_data(raw_data, dataset_name)  # Use base dataset name
        elif category == "cause-effect":
            examples = process_cause_effect_data(raw_data, dataset_name)  # Use base dataset name
        elif category == "scm":
            examples = process_scm_data(raw_data, dataset_name)  # SCM returns a dictionary grouped by query type
            # Flatten SCM examples by query type into combined_examples
            for query_type, query_examples in examples.items():
                combined_examples.extend(query_examples)
        else:
            raise ValueError(f"Unsupported category: {category}")

        # For non-SCM datasets, directly append examples
        if category != "scm":
            combined_examples.extend(examples)

    return combined_examples


def process_counterfactual_data(dataset, dataset_name):
    """
    Process counterfactual reasoning datasets (e.g., TimeTravel, bAbI Task).

    Args:
        dataset (Dataset): The dataset to process.
        dataset_name (str): Name of the dataset (e.g., "wza/TimeTravel", "facebook/babi_qa").

    Returns:
        list[dict]: Processed examples in a standardized format (without prompts).
    """
    examples = []

    if dataset_name == "wza/TimeTravel":
        for item in dataset:
            # Skip malformed examples
            if not all(key in item for key in ["premise", "counterfactual", "original_ending", "edited_ending"]):
                print(f"Skipping malformed example: {item}")
                continue

            examples.append({
                "input": counterfactual_prompt(item, dataset_name),
                "output": item["edited_ending"],
                "metadata": {"dataset": dataset_name, "story_id": item["story_id"]},
            })

    elif dataset_name.startswith("facebook/babi_qa"):
        for item in dataset:
            story = item["story"]
            story_text = story["text"]
            story_type = story["type"]
            answers = story["answer"]

            for i, line_type in enumerate(story_type):
                if line_type == 1:  # Line is a question
                    question = story_text[i]
                    answer = answers[i]
                    context = "\n".join(story_text[:i])  # Context includes lines before the question

                    # Skip malformed examples
                    if not context.strip() or not question.strip() or not answer.strip():
                        print(f"Skipping malformed example: {item}")
                        continue

                    examples.append({
                        "input": counterfactual_prompt({
                            "context": context,
                            "question": question
                        }, dataset_name),
                        "output": answer,
                        "metadata": {"dataset": dataset_name, "question_id": story["id"][i]},
                    })

    else:
        raise ValueError(f"Unsupported dataset for counterfactual reasoning: {dataset_name}")

    return examples


def process_cause_effect_data(dataset, dataset_name):
    """
    Process cause-effect datasets (e.g., CommonsenseQA, balanced-copa).

    Args:
        dataset (Dataset): The dataset to process.
        dataset_name (str): Name of the dataset (e.g., "tau/commonsense_qa", "pkavumba/balanced-copa").

    Returns:
        list[dict]: Processed examples in a standardized format.
    """
    examples = []

    if dataset_name == "tau/commonsense_qa":
        for item in dataset:
            if not isinstance(item.get("choices"), dict) or \
                    not all(key in item["choices"] for key in ["label", "text"]) or \
                    not isinstance(item["choices"]["label"], list) or \
                    not isinstance(item["choices"]["text"], list):
                print(f"Skipping malformed 'choices' field: {item}")
                continue

            if not all(key in item for key in ["question", "choices", "answerKey"]):
                print(f"Skipping malformed example: {item}")
                continue

            choices = dict(zip(item["choices"]["label"], item["choices"]["text"]))
            correct_answer_label = item["answerKey"]

            if correct_answer_label not in choices:
                print(f"Skipping example with invalid answerKey: {item}")
                continue

            examples.append({
                "input": cause_effect_prompt(item, dataset_name),
                "output": correct_answer_label,
                "metadata": {"dataset": dataset_name, "question_id": item.get("id", "unknown")},
            })

    elif dataset_name == "pkavumba/balanced-copa":
        for item in dataset:
            if not all(key in item for key in ["premise", "question", "choice1", "choice2", "label"]):
                print(f"Skipping malformed example: {item}")
                continue

            # Convert numeric label (0 or 1) to corresponding choice (1 or 2)
            correct_answer = "1" if item["label"] == 0 else "2"

            examples.append({
                "input": cause_effect_prompt(item, dataset_name),
                "output": correct_answer,
                "metadata": {"dataset": dataset_name, "question_id": item.get("id", "unknown")},
            })

    elif dataset_name == "allenai/art":
        for item in dataset:
            if not all(
                    key in item for key in ["observation_1", "observation_2", "hypothesis_1", "hypothesis_2", "label"]):
                print(f"Skipping malformed example: {item}")
                continue

            # Convert label to string ('1' or '2') for consistency
            label = str(item["label"])
            if label not in ["1", "2"]:
                print(f"Skipping example with invalid label: {item}")
                continue

            examples.append({
                "input": cause_effect_prompt(item, dataset_name),
                "output": label,
                "metadata": {"dataset": dataset_name, "question_id": item.get("id", "unknown")},
            })

    else:
        raise ValueError(f"Unsupported dataset for cause-effect reasoning: {dataset_name}")

    return examples

def process_scm_data(dataset, dataset_name):
    """
    Process SCM datasets (e.g., CLADDER) with multi-task support based on query types.

    Args:
        dataset (Dataset): The dataset to process.
        dataset_name (str): Name of the dataset (e.g., "causal-nlp/cladder").

    Returns:
        dict[str, list[dict]]: Dictionary mapping query types to processed examples.
    """
    examples_by_query_type = {}

    if dataset_name == "causal-nlp/CLadder":
        for item in dataset:
            # Ensure required fields exist
            if not all(key in item for key in ["prompt", "label", "query_type"]):
                print(f"Skipping malformed example: {item}")
                continue

            query_type = item["query_type"]  # Treat each query type as a separate task
            if query_type not in examples_by_query_type:
                examples_by_query_type[query_type] = []

            examples_by_query_type[query_type].append({
                "input": scm_prompt(item, dataset_name),
                "output": item["label"].strip().lower(),  # Normalize label to 'yes' or 'no'
                "metadata": {"dataset": dataset_name, "query_type": query_type, "rung": item.get("rung", "unknown")},
            })
    else:
        raise ValueError(f"Unsupported dataset for SCM reasoning: {dataset_name}")

    return examples_by_query_type

