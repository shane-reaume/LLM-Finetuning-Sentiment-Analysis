import argparse
import os
import requests
from huggingface_hub import HfApi

def update_model_card(repo_name, model_card_path=None, model_type="standard"):
    """
    Update the model card for a model on Hugging Face Hub
    
    Args:
        repo_name (str): Name of the repository on Hugging Face Hub
        model_card_path (str, optional): Path to model card markdown file
        model_type (str): Type of model - "standard" or "improved"
    """
    # Initialize Hugging Face API
    api = HfApi()
    
    # If no model card path provided, create one
    if model_card_path is None:
        model_card = create_model_card(repo_name, model_type)
        model_card_path = "temp_model_card.md"
        with open(model_card_path, "w") as f:
            f.write(model_card)
    else:
        # Read the model card from file
        with open(model_card_path, "r") as f:
            model_card = f.read()
    
    # Upload the model card
    api.upload_file(
        path_or_fileobj=model_card_path,
        path_in_repo="README.md",
        repo_id=repo_name,
        repo_type="model",
    )
    
    # Clean up temporary file if created
    if model_card_path == "temp_model_card.md" and os.path.exists(model_card_path):
        os.remove(model_card_path)
    
    print(f"Model card updated for {repo_name}")

def create_model_card(repo_name, model_type="standard"):
    """Create a detailed model card"""
    
    # Performance metrics based on model type
    if model_type == "improved":
        accuracy = "86.50%"
        f1_score = "0.8672"
        precision = "84.21%"
        recall = "89.47%"
        model_description = "This is an improved version of the sentiment analysis model, fine-tuned with additional challenging examples to handle difficult cases like negation, sarcasm, and subtle expressions."
        model_version = "2.0"
        training_details = "The model was trained on the IMDB dataset augmented with challenging examples specifically designed to improve performance on difficult sentiment analysis cases."
    else:
        accuracy = "84.00%"
        f1_score = "0.8462"
        precision = "81.48%"
        recall = "88.00%"
        model_description = "This is a fine-tuned DistilBERT model for sentiment analysis on the IMDB movie reviews dataset."
        model_version = "1.0"
        training_details = "The model was trained on the IMDB dataset using supervised fine-tuning."
    
    # Create the model card with proper escaping for code examples
    model_card = f"""---
language: en
license: mit
library_name: transformers
tags:
- sentiment-analysis
- text-classification
- pytorch
- distilbert
- imdb
datasets:
- imdb
metrics:
- accuracy
- f1
- precision
- recall
model-index:
- name: {repo_name.split('/')[-1]}
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: IMDB
      type: imdb
      split: test
    metrics:
    - type: accuracy
      value: {accuracy.replace('%', '')}
      name: Accuracy
    - type: f1
      value: {f1_score}
      name: F1 Score
---

# Sentiment Analysis Model v{model_version}

{model_description}

## Model Details

- **Model Type:** DistilBERT (fine-tuned)
- **Task:** Binary Sentiment Classification (Positive/Negative)
- **Training Data:** IMDB Movie Reviews Dataset
- **Language:** English
- **License:** MIT
- **Version:** {model_version}

## Performance

| Metric | Value |
|--------|-------|
| Accuracy | {accuracy} |
| F1 Score | {f1_score} |
| Precision | {precision} |
| Recall | {recall} |

## Training Details

{training_details}

### Training Hyperparameters

- Learning Rate: 2e-5
- Batch Size: 16 (effective batch size: 32 with gradient accumulation)
- Epochs: 3
- Optimizer: AdamW with weight decay
- Mixed Precision: FP16

## Usage

### Direct Use with Pipeline

```python
from transformers import pipeline

# Load the model
sentiment = pipeline("sentiment-analysis", model="{repo_name}")

# Analyze text
result = sentiment("I really enjoyed this movie!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]

# Batch processing
texts = [
    "This movie was absolutely amazing, I loved every minute of it!",
    "The acting was terrible and the plot made no sense at all."
]
results = sentiment(texts)
for i, (text, result) in enumerate(zip(texts, results)):
    print(f"Text: {{text}}")
    print(f"Sentiment: {{result['label']}}, Score: {{result['score']:.4f}}")
```

### Loading Model Directly

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "{repo_name}"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare text
text = "I really enjoyed this movie!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

# Get prediction
with torch.no_grad():
    outputs = model(**inputs)
    
# Process outputs
probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
prediction = torch.argmax(probabilities, dim=-1).item()
confidence = probabilities[0][prediction].item()

# Map prediction to label (0: negative, 1: positive)
sentiment_label = "POSITIVE" if prediction == 1 else "NEGATIVE"
print(f"Sentiment: {{sentiment_label}}, Confidence: {{confidence:.4f}}")
```

## Limitations

- The model is trained primarily on movie reviews and may not perform as well on other domains.
- The model may struggle with certain types of text:
  - Sarcasm and irony
  - Mixed sentiment expressions
  - Subtle negative expressions
  - Complex negations

## Citation

If you use this model in your research, please cite:

```
@misc{{sentiment-analysis-model,
  author = {{Your Name}},
  title = {{Sentiment Analysis Model based on DistilBERT}},
  year = {{2023}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/{repo_name}}}}}
}}
```
"""
    return model_card

def main():
    parser = argparse.ArgumentParser(description="Update model card on Hugging Face Hub")
    parser.add_argument("--repo_name", type=str, required=True,
                        help="Repository name on Hugging Face Hub (e.g., username/model-name)")
    parser.add_argument("--model_card", type=str, default=None,
                        help="Path to model card markdown file")
    parser.add_argument("--improved", action="store_true",
                        help="Use improved model card template")
    args = parser.parse_args()
    
    model_type = "improved" if args.improved else "standard"
    update_model_card(args.repo_name, args.model_card, model_type)

if __name__ == "__main__":
    main()
