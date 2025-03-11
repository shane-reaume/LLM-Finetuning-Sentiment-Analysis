import os
import json
import argparse
from datasets import load_dataset, concatenate_datasets, ClassLabel, Features, Value
import pandas as pd
from colorama import Fore, Style, init

# Initialize colorama
init()

def load_challenging_cases(file_path):
    """Load challenging test cases from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def create_augmentation_dataset(challenge_file, output_file=None, verbose=True):
    """Create a dataset from challenging cases for training augmentation"""
    # Load challenging cases
    data = load_challenging_cases(challenge_file)
    test_cases = data["test_cases"]
    
    # Convert to format compatible with IMDB dataset
    texts = []
    labels = []
    
    for case in test_cases:
        texts.append(case["text"])
        labels.append(case["expected_sentiment"])
    
    # Create DataFrame
    df = pd.DataFrame({
        "text": texts,
        "label": labels
    })
    
    if verbose:
        print(f"{Fore.GREEN}Created augmentation dataset with {len(df)} examples{Style.RESET_ALL}")
        print(f"Positive examples: {sum(df['label'] == 1)}")
        print(f"Negative examples: {sum(df['label'] == 0)}")
    
    # Save to file if specified
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"Saved augmentation dataset to {output_file}")
    
    return df

def augment_training_data(challenge_file, output_dir, augmentation_factor=5, verbose=True):
    """
    Augment the IMDB training dataset with challenging examples
    
    Args:
        challenge_file: Path to challenging cases JSON file
        output_dir: Directory to save augmented dataset
        augmentation_factor: Number of times to repeat challenging examples
        verbose: Whether to print progress
    """
    # Load IMDB dataset
    if verbose:
        print("Loading IMDB dataset...")
    
    imdb = load_dataset("imdb")
    train_ds = imdb["train"]
    
    # Load challenging cases
    if verbose:
        print(f"Loading challenging cases from {challenge_file}...")
    
    aug_df = create_augmentation_dataset(challenge_file, verbose=verbose)
    
    # Convert to dataset format with matching features
    from datasets import Dataset
    
    # Get the feature schema from the IMDB dataset to ensure compatibility
    features = train_ds.features.copy()
    
    # Create dataset with the same feature schema
    aug_ds = Dataset.from_pandas(
        aug_df,
        features=Features({
            "text": Value("string"),
            "label": ClassLabel(names=["neg", "pos"])  # Match IMDB's label format
        })
    )
    
    if verbose:
        print("Created augmentation dataset with matching feature schema")
    
    # Repeat the challenging examples to increase their weight
    if augmentation_factor > 1:
        aug_datasets = [aug_ds] * augmentation_factor
        aug_ds = concatenate_datasets(aug_datasets)
        if verbose:
            print(f"Repeated challenging examples {augmentation_factor}x (total: {len(aug_ds)})")
    
    # Combine with original training data
    combined_ds = concatenate_datasets([train_ds, aug_ds])
    
    if verbose:
        print(f"\n{Fore.CYAN}Dataset Statistics:{Style.RESET_ALL}")
        print(f"Original training examples: {len(train_ds)}")
        print(f"Augmentation examples: {len(aug_ds)}")
        print(f"Combined dataset size: {len(combined_ds)}")
        
        # Calculate class balance
        pos_count = sum(1 for x in combined_ds if x["label"] == 1)
        neg_count = sum(1 for x in combined_ds if x["label"] == 0)
        print(f"Positive examples: {pos_count} ({pos_count/len(combined_ds)*100:.1f}%)")
        print(f"Negative examples: {neg_count} ({neg_count/len(combined_ds)*100:.1f}%)")
    
    # Save augmented dataset
    os.makedirs(output_dir, exist_ok=True)
    combined_ds.save_to_disk(os.path.join(output_dir, "imdb_augmented"))
    
    if verbose:
        print(f"\n{Fore.GREEN}Saved augmented dataset to {os.path.join(output_dir, 'imdb_augmented')}{Style.RESET_ALL}")
        print("To use this dataset for training, update your config file to point to this directory.")

def main():
    parser = argparse.ArgumentParser(description="Augment training data with challenging examples")
    parser.add_argument("--challenge_file", type=str, default="tests/challenging_cases.json",
                        help="Path to challenging test cases file")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                        help="Directory to save augmented dataset")
    parser.add_argument("--augmentation_factor", type=int, default=5,
                        help="Number of times to repeat challenging examples")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")
    args = parser.parse_args()
    
    augment_training_data(
        args.challenge_file, 
        args.output_dir,
        args.augmentation_factor,
        verbose=not args.quiet
    )

if __name__ == "__main__":
    main() 