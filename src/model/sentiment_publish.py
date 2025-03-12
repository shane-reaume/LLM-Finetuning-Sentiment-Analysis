import argparse
import os
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.utils.config_utils import load_config

def publish_model(model_dir, repo_name, description=None, version=None):
    """
    Publish fine-tuned model to Hugging Face Hub
    
    Args:
        model_dir (str): Path to model directory
        repo_name (str): Name for the repository on Hugging Face Hub
        description (str, optional): Description for the model
        version (str, optional): Version tag for the model
    """
    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Create default description if not provided
    if description is None:
        description = """
        # Sentiment Analysis Model
        
        This is a fine-tuned DistilBERT model for sentiment analysis on the IMDB dataset.
        
        ## Performance
        
        - Accuracy: 84.00%
        - F1 Score: 0.8462
        - Precision: 81.48%
        - Recall: 88.00%
        
        ## Usage
        
        ```python
        from transformers import pipeline
        
        sentiment = pipeline("sentiment-analysis", model="{repo_name}")
        result = sentiment("I really enjoyed this movie!")
        print(result)
        ```
        """.format(repo_name=repo_name)
    
    # Add model metadata for proper inference API support
    model_info = {
        "library_name": "transformers",
        "task": "text-classification",
        "tags": ["sentiment-analysis", "text-classification", "pytorch", "distilbert", "imdb"],
        "pipeline_tag": "text-classification",
        "language": "en",
        "license": "mit",
        "base_model": "distilbert-base-uncased"
    }
    
    # Add version tag if provided
    if version:
        model_info["version"] = version
        
    # Save model info to model directory
    with open(os.path.join(model_dir, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)
    
    # Push to Hub
    model.push_to_hub(
        repo_name, 
        use_auth_token=True,
        commit_message=f"Upload model {'v' + version if version else ''}",
        metadata=model_info
    )
    
    tokenizer.push_to_hub(
        repo_name, 
        use_auth_token=True,
        commit_message=f"Upload tokenizer {'v' + version if version else ''}"
    )
    
    print(f"Model successfully published to: https://huggingface.co/{repo_name}")
    print("The model is now properly configured for the Hugging Face Inference API")
    
def main():
    parser = argparse.ArgumentParser(description="Publish model to Hugging Face Hub")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Path to model directory")
    parser.add_argument("--repo_name", type=str, required=True,
                        help="Repository name on Hugging Face Hub (e.g., username/model-name)")
    parser.add_argument("--description", type=str, default=None,
                        help="Description for the model")
    parser.add_argument("--config", type=str, default="config/sentiment_analysis.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--version", type=str, default=None,
                        help="Version tag for the model (e.g., '1.0')")
    parser.add_argument("--improved", action="store_true",
                        help="Use the improved model variant")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine model path
    if args.improved:
        model_path = "models/sentiment_improved"
    else:
        # Use provided model_dir or default from config
        model_path = args.model_dir if args.model_dir else config["model"]["save_dir"]
    
    # Publish model
    publish_model(model_path, args.repo_name, args.description, args.version)

if __name__ == "__main__":
    main()
