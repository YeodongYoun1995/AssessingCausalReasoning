def counterfactual_prompt(example, dataset_name):
    """
    Construct a counterfactual reasoning prompt with instructions.

    Args:
        example (dict): Example data containing necessary fields.
        dataset_name (str): Name of the dataset (e.g., "wza/TimeTravel", "facebook/babi_qa").

    Returns:
        str: Fully formatted zero-shot prompt.
    """
    if dataset_name == "wza/TimeTravel":
        # Dataset-specific formatting for TimeTravel
        instruction = "You are tasked with counterfactual reasoning. Only return the edited ending that follows the given counterfactual condition. Strictly avoid repeating the premise or adding explanations."
        premise = example["premise"]
        counterfactual = example["counterfactual"]
        original_ending = example["original_ending"]

        query = (
            f"Premise: {premise}\n"
            f"Counterfactual: {counterfactual}\n"
            f"Original Ending: {original_ending}\n"
            f"Edited Ending:"
        )

    elif dataset_name.startswith("facebook/babi_qa"):
        # Dataset-specific formatting for bAbI Task
        instruction = "Only provide the correct answer without repeating the question or adding explanations. Strictly avoid repeating the premise or adding explanations."
        context_text = example.get("context", "No context provided.")
        question = example.get("question", "No question provided.")

        query = (
            f"Context: {context_text}\n"
            f"Question: {question}\n"
            f"Answer:"
        )

    else:
        raise ValueError(f"Unsupported dataset for counterfactual reasoning: {dataset_name}")

    # Combine instruction and query for the zero-shot prompt
    return f"{instruction}\n\n{query}"


def cause_effect_prompt(example, dataset_name):
    """
    Construct a cause-effect reasoning prompt with instructions.

    Args:
        example (dict): Example data containing necessary fields.
        dataset_name (str): Name of the dataset (e.g., "tau/commonsense_qa", "pkavumba/balanced-copa").

    Returns:
        str: Fully formatted zero-shot prompt.
    """
    if dataset_name == "tau/commonsense_qa":
        instruction = (
            "You are tasked with identifying cause-effect relationships. "
            "Only generate the correct answer choice among the provided multiple choices. "
            "Strictly avoid repeating the question and choices or adding explanations."
        )
        if not all(key in example["choices"] for key in ["label", "text"]):
            raise ValueError(f"Malformed 'choices' structure: {example['choices']}")

        choices = dict(zip(example["choices"]["label"], example["choices"]["text"]))
        formatted_choices = ", ".join(f"{label}: {text}" for label, text in choices.items())
        question = example["question"]

        dataset_prompt = f"Question: {question}\nChoices: {formatted_choices}.\nAnswer:"

    elif dataset_name == "pkavumba/balanced-copa":
        instruction = (
            "You are tasked with identifying the correct causal relationship. "
            "Choose between the two provided options strictly based on the given premise."
        )
        premise = example["premise"]
        choice1 = example["choice1"]
        choice2 = example["choice2"]

        dataset_prompt = (
            f"Premise: {premise}\n"
            f"Question: {example['question']}\n"
            f"Choices:\n1: {choice1}\n2: {choice2}\nAnswer:"
        )


    elif dataset_name == "allenai/art":
        # Instructions specific to ART dataset
        instruction = (
            "You are tasked with selecting the more plausible hypothesis based on two observations."
            " Only return '1' or '2' for your choice without explanations."
        )

        obs_1 = example["observation_1"]
        obs_2 = example["observation_2"]
        hyp_1 = example["hypothesis_1"]
        hyp_2 = example["hypothesis_2"]
        dataset_prompt = (
            f"Observation 1: {obs_1}\n"
            f"Observation 2: {obs_2}\n"
            f"Hypothesis 1: {hyp_1}\n"
            f"Hypothesis 2: {hyp_2}\n"
            f"Which hypothesis is more plausible? (1 or 2)\nAnswer:"
        )
    else:
        raise ValueError(f"Unsupported dataset for cause-effect reasoning: {dataset_name}")

    # Combine instruction and dataset-specific prompt
    return f"{instruction}\n\n{dataset_prompt}"

def scm_prompt(example, dataset_name):
    """
    Construct an SCM reasoning prompt with instructions.

    Args:
        example (dict): Example data containing necessary fields.
        dataset_name (str): Name of the dataset (e.g., "causal-nlp/cladder").

    Returns:
        str: Fully formatted zero-shot prompt.
    """
    if dataset_name == "causal-nlp/CLadder":
        # Instruction for SCM task with multi-query
        instruction = (
            f"You are tasked with answering causal inference questions. "
            f"The query type is '{example['query_type']}'. Answer 'yes' or 'no' based on the provided context and question."
        )

        # Extract fields
        prompt = example["prompt"]

        dataset_prompt = f"Context and Question: {prompt}\nAnswer:"
    else:
        raise ValueError(f"Unsupported dataset for SCM reasoning: {dataset_name}")

    return f"{instruction}\n\n{dataset_prompt}"