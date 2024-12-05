import argparse
from eval import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Causal Reasoning Evaluation")

    # Hyperparameters
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search (default: 1, greedy decoding)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for generation (default: 1.0)")

    # Model selection
    parser.add_argument("--model", type=str, required=True, help="Model name to be tested (e.g., gpt-neo, flan-t5)")

    # Dataset category and name
    parser.add_argument("--category", type=str, required=True, help="Dataset category for evaluation (e.g., counterfactual, cause-effect, scm)")
    parser.add_argument("--dataset_name", type=str, required=False, help="Specific dataset name within the category (e.g., wza/TimeTravel:supervised_small, tau/commonsense_qa)")

    # Metric selection
    parser.add_argument("--metric", type=str, required=False, choices=["accuracy", "f1", "bleu", "rouge"],
                        help="Metric for evaluation (e.g., accuracy, f1, bleu, rouge). If not specified, all metrics will be evaluated.")

    # GPU selection
    parser.add_argument("--gpu", type=int, default=0, help="GPU device to use (default: 0)")

    args = parser.parse_args()

    # Display selected arguments
    print(f"Starting evaluation with the following settings:")
    print(f"  Model: {args.model}")
    print(f"  Category: {args.category}")
    if args.dataset_name:
        print(f"  Dataset: {args.dataset_name}")
    print(f"  Metric: {args.metric or 'all available metrics'}")
    print(f"  Num Beams: {args.num_beams}")
    print(f"  Temperature: {args.temperature}")
    print(f"  GPU: {args.gpu}")

    # Run evaluation
    evaluate_model(
        model_name=args.model,
        category=args.category,
        metric=args.metric,
        dataset_name=args.dataset_name,
        gpu=args.gpu,
        num_beams=args.num_beams,
        temperature=args.temperature
    )

if __name__ == "__main__":
    main()