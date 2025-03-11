import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

def load_and_preprocess_data(config):
    """
    Load and preprocess a text classification dataset from Hugging Face's datasets library.
    
    Args:
        config (dict): Configuration dictionary with data parameters
        
    Returns:
        tuple: preprocessed train and validation datasets ready for training
    """
    # Load tokenizer based on model name
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    
    # Check if we should use the augmented dataset
    use_augmented = config["data"].get("use_augmented_dataset", False)
    
    if use_augmented:
        print(f"Using augmented dataset from {config['data']['augmented_dataset_path']}")
        # Load the augmented training dataset from disk
        train_dataset_raw = load_from_disk(config["data"]["augmented_dataset_path"])
        
        # Load the regular dataset for validation
        regular_dataset = load_dataset(
            config["data"]["dataset_name"],
            cache_dir=config["data"]["cache_dir"]
        )
        validation_dataset_raw = regular_dataset[config["data"]["validation_split"]]
    else:
        # Load dataset from Hugging Face
        dataset = load_dataset(
            config["data"]["dataset_name"],
            cache_dir=config["data"]["cache_dir"]
        )
        
        # Use regular splits
        train_dataset_raw = dataset[config["data"]["train_split"]]
        validation_dataset_raw = dataset[config["data"]["validation_split"]]
    
    # Define preprocessing function
    def preprocess_function(examples):
        # Tokenize the texts
        result = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=config["data"]["max_length"]
        )
        
        # Map labels for classification
        result["labels"] = examples["label"]
        return result
    
    # Apply preprocessing
    train_dataset = train_dataset_raw.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset_raw.column_names
    )
    
    validation_dataset = validation_dataset_raw.map(
        preprocess_function,
        batched=True,
        remove_columns=validation_dataset_raw.column_names
    )
    
    # Set the format for PyTorch
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    validation_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    # Print dataset statistics
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")
    
    return train_dataset, validation_dataset

def create_test_examples(config, num_examples=50):
    """
    Create a small set of test examples for evaluation and testing
    
    Args:
        config (dict): Configuration dictionary with data parameters
        num_examples (int): Number of examples to extract
        
    Returns:
        list: List of test examples with text and expected label
    """
    try:
        dataset = load_dataset(
            config["data"]["dataset_name"],
            cache_dir=config["data"]["cache_dir"]
        )
        
        # Take a subset of the validation set for testing
        test_subset = dataset[config["data"]["validation_split"]].select(range(num_examples))
        
        # Format as a list of dictionaries
        test_examples = [
            {"text": example["text"], "label": example["label"]}
            for example in test_subset
        ]
        
        return test_examples  # Make sure to return the list
    except Exception as e:
        print(f"Error creating test examples: {e}")
        # Return empty list instead of None
        return []
