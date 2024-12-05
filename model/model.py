import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

class BaseModel:
    """
    Base class for models to ensure consistent interface.
    """

    def __init__(self, model_name, model_type="causal", gpu=0):
        """
        Initialize the model and tokenizer.

        Args:
            model_name (str): The name of the Hugging Face model (e.g., "EleutherAI/gpt-neo-2.7B").
            model_type (str): The type of model ("causal" for causal LM, "seq2seq" for seq2seq LM).
            gpu (int): The GPU device to use (default: 0).
        """
        self.model_name = model_name
        self.model_type = model_type

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set pad_token_id if not already set
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.model.resize_token_embeddings(len(self.tokenizer))

        if model_type == "causal":
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        elif model_type == "seq2seq":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Set the device based on the GPU argument
        if torch.cuda.is_available() and gpu >= 0:
            self.device = torch.device(f"cuda:{gpu}")
            print(f"Using GPU {gpu} for inference.")
        else:
            self.device = torch.device("cpu")
            print("Using CPU for inference.")

        self.model.to(self.device)  # Move model to the specified device

    def generate(self, prompt, max_new_tokens=512, num_beams=1, temperature=1.0):
        """
        Generate a response from the model.

        Args:
            prompt (str): Input prompt.
            max_new_tokens (int): Maximum number of tokens to generate.
            num_beams (int): Number of beams for beam search (optional).
            temperature (float): Sampling temperature (optional).

        Returns:
            str: Generated response.
        """
        # Tokenize the input prompt
        inputs = self.tokenizer(
            prompt,
            truncation=True,
            return_tensors="pt",
            padding=True,  # Add padding if necessary
        )

        # Move inputs to the appropriate device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Generate response
        response_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            temperature=temperature,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode the response
        response = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
        return response


def load_model(model_name, gpu=0):
    """
    Load a specified model based on its name.

    Args:
        model_name (str): The name of the model to load (e.g., "gpt-neo", "flan-t5", "llama").
        gpu (int): The GPU device to use (default: 0).

    Returns:
        BaseModel: An instance of the loaded model.
    """
    # Map model names to Hugging Face checkpoints
    MODEL_MAP = {
        "gpt-neo": {"name": "EleutherAI/gpt-neo-2.7B", "type": "causal"},
        "gpt-j": {"name": "EleutherAI/gpt-j-6B", "type": "causal"},
        "flan-t5": {"name": "google/flan-t5-large", "type": "seq2seq"},
        "flan-t5-xxl": {"name": "google/flan-t5-xxl", "type": "seq2seq"},
        "bart": {"name": "facebook/bart-large", "type": "seq2seq"},
        "gpt2": {"name": "gpt2", "type": "causal"},  # Smaller GPT-2 model
        "llama2": {"name": "meta-llama/Llama-2-7b-hf", "type": "causal"},  # Add LLaMA 2 checkpoint
    }

    if model_name not in MODEL_MAP:
        raise ValueError(f"Unsupported model name: {model_name}")

    model_info = MODEL_MAP[model_name]
    return BaseModel(model_info["name"], model_info["type"], gpu=gpu)