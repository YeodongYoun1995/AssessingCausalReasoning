from sklearn.metrics import f1_score as sklearn_f1_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def calculate_metric(predictions, references, metric):
    """
    Calculate the specified evaluation metric.

    Args:
        predictions (list[str]): Model-generated predictions.
        references (list[str]): Ground-truth references.
        metric (str): The evaluation metric to calculate ("accuracy", "f1", "bleu", "rouge").

    Returns:
        float: The computed metric score.
    """
    if metric == "accuracy":
        return accuracy(predictions, references)
    elif metric == "f1":
        return f1_score(predictions, references)
    elif metric == "bleu":
        return bleu_score(predictions, references)
    elif metric == "rouge":
        return rouge_score(predictions, references)
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def accuracy(predictions, references):
    """
    Compute accuracy between predictions and references.

    Args:
        predictions (list[str]): Model-generated predictions.
        references (list[str]): Ground-truth references.

    Returns:
        float: Accuracy score.
    """
    correct = sum([1 for p, r in zip(predictions, references) if p == r])
    return correct / len(references)


def f1_score(predictions, references):
    """
    Compute the F1 score between predictions and references.

    Args:
        predictions (list[str]): Model-generated predictions.
        references (list[str]): Ground-truth references.

    Returns:
        float: F1 score.
    """
    binary_predictions = [1 if p == r else 0 for p, r in zip(predictions, references)]
    binary_references = [1] * len(references)  # All ground truths are "1" (correct)
    return sklearn_f1_score(binary_references, binary_predictions)


def bleu_score(predictions, references):
    """
    Compute the BLEU score for text generation tasks.

    Args:
        predictions (list[str]): Model-generated predictions.
        references (list[str]): Ground-truth references.

    Returns:
        float: Average BLEU score.
    """
    smoothie = SmoothingFunction().method4
    scores = [
        sentence_bleu([r.split()], p.split(), smoothing_function=smoothie)
        for p, r in zip(predictions, references)
    ]
    return sum(scores) / len(scores)


def rouge_score(predictions, references):
    """
    Compute the ROUGE-L score for text generation tasks.

    Args:
        predictions (list[str]): Model-generated predictions.
        references (list[str]): Ground-truth references.

    Returns:
        float: Average ROUGE-L score.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(predictions, references)
    ]
    return sum(scores) / len(scores)