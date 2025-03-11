import os
import argparse
import yaml
from colorama import Fore, Style, init

from src.utils.config_utils import load_config
from src.data.sentiment_augment_training import augment_training_data
from src.model.sentiment_train import train_model

# Initialize colorama
init()

def train_with_challenging_cases(config_path, challenge_file, augmentation_factor=5):
    """
    Train a sentiment analysis model with challenging cases
    
    Args:
        config_path: Path to configuration file
        challenge_file: Path to challenging cases file
        augmentation_factor: Number of times to repeat challenging examples
    """
    # Step 1: Load configuration
    print(f"{Fore.CYAN}Loading configuration from {config_path}...{Style.RESET_ALL}")
    config = load_config(config_path)
    
    # Step 2: Create augmented dataset
    print(f"\n{Fore.CYAN}Creating augmented dataset with challenging cases...{Style.RESET_ALL}")
    augment_training_data(
        challenge_file=challenge_file,
        output_dir=os.path.dirname(config["data"]["augmented_dataset_path"]),
        augmentation_factor=augmentation_factor
    )
    
    # Step 3: Update configuration to use augmented dataset
    print(f"\n{Fore.CYAN}Updating configuration to use augmented dataset...{Style.RESET_ALL}")
    config["data"]["use_augmented_dataset"] = True
    
    # Step 4: Train model with augmented dataset
    print(f"\n{Fore.CYAN}Training model with augmented dataset...{Style.RESET_ALL}")
    model, tokenizer = train_model(config)
    
    print(f"\n{Fore.GREEN}Training complete! Model saved to {config['model']['save_dir']}{Style.RESET_ALL}")
    print(f"To evaluate the model on challenging cases, run:")
    print(f"python -m src.model.sentiment_challenge_test --model_dir {config['model']['save_dir']}")

def main():
    parser = argparse.ArgumentParser(description="Train sentiment model with challenging cases")
    parser.add_argument("--config", type=str, default="config/sentiment_analysis.yaml",
                        help="Path to configuration file")
    parser.add_argument("--challenge_file", type=str, default="tests/challenging_cases.json",
                        help="Path to challenging test cases file")
    parser.add_argument("--augmentation_factor", type=int, default=5,
                        help="Number of times to repeat challenging examples")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save the model (overrides config)")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override save directory if provided
    if args.save_dir:
        config["model"]["save_dir"] = args.save_dir
        
        # Create a new config file with the updated save directory
        config_dir = os.path.dirname(args.config)
        config_name = os.path.basename(args.config).split(".")[0]
        new_config_path = os.path.join(config_dir, f"{config_name}_augmented.yaml")
        
        with open(new_config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Created new config file with updated save directory: {new_config_path}")
        args.config = new_config_path
    
    train_with_challenging_cases(
        args.config,
        args.challenge_file,
        args.augmentation_factor
    )

if __name__ == "__main__":
    main() 