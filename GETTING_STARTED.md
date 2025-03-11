# Getting Started with LLM-Finetuning-Sentiment-Analysis

**Sentiment Analysis** - A beginner-friendly demo project for fine-tuning and testing a text classification model

## 🔍 Project Overview

This demo project demonstrates:

* **Model Architecture**: 
  * Fine-tuning an encoder-based **DistilBERT** model for binary text classification
  * Transfer learning approach using a pre-trained language model with a classification head
  * Tokenizer-based preprocessing with fixed sequence length (128 tokens)

* **Training Methodology**:
  * Binary classification on the IMDB movie reviews dataset (positive/negative sentiment)
  * Supervised fine-tuning with cross-entropy loss
  * Mixed-precision training (FP16) for memory efficiency
  * AdamW optimizer with learning rate of 2e-5 and weight decay

* **Testing Framework**:
  * **Unit Tests**: Validating model loading and architecture
  * **Functional Tests**: Verifying correct predictions on known examples
  * **Performance Tests**: Ensuring accuracy and inference speed meet thresholds
  * **Edge Case Tests**: Testing model behavior with unusual inputs
  * **Batch Processing Tests**: Validating efficient batch prediction capabilities

* **Evaluation Metrics**:
  * Primary metrics: Accuracy and F1 score
  * Secondary metrics: Precision, recall, and confidence thresholds
  * Performance benchmarking: Inference latency measurements

## 🎓 Demo Model

If you want to use a trained version of this sentiment analysis model before learning how to train your own, it is available on Hugging Face:
[https://huggingface.co/shane-reaume/imdb-sentiment-analysis](https://huggingface.co/shane-reaume/imdb-sentiment-analysis)

You can try it directly in your browser or use it in your code with the Hugging Face API:

```python
from transformers import pipeline

# Load the model
sentiment = pipeline("sentiment-analysis", model="shane-reaume/imdb-sentiment-analysis")

# Make predictions
result = sentiment("I really enjoyed this movie!")
print(result)
```

## 📋 Common Prerequisites

- **Python 3.13**
- **Git** for version control
- **GPU with 8GB+ VRAM** recommended for training (CPU can be used but will be very slow)
- **Basic Python knowledge** (No ML experience required)

## 🔧 Initial Setup

### Step 1: Clone the repository and set up environment

```bash
git clone https://github.com/your-repo/LLM-Finetuning-Sentiment-Analysis.git
cd LLM-Finetuning-Sentiment-Analysis
chmod +x setup_env.sh  # Only needs to be done once
./setup_env.sh
```

This will:

- Create a virtual environment in the `venv` directory
- Install all dependencies
- Create necessary project directories

### Step 2: Activate the virtual environment
(if not already activated by the script)

```bash
source venv/bin/activate
```

## Sentiment Analysis Project

This project is a sentiment analysis model using movie reviews. Here's how to use it:

### Step 1: Create Test Examples

Generate a set of test examples that will be used for consistent evaluation:

```bash
python -m src.data.sentiment_create_test_set
```

This creates a JSON file with test examples at `data/processed/sentiment_test_examples.json`.

### Step 2: Train the Model

Start the training process:

```bash
python -m src.model.sentiment_train
```

This will:

- Download the IMDB dataset
- Load the DistilBERT model
- Fine-tune it for sentiment analysis
- Save the model to `models/sentiment`

Training should take approximately 1-2 hours on an 8GB GPU.

### Step 3: Run Tests

Evaluate the model's performance:

```bash
pytest tests/test_sentiment_model.py -v
```

This will run a suite of tests to check that:

- The model can be loaded correctly
- Predictions work as expected
- The model meets performance requirements

### Step 4: Try the Interactive Demo

Once your model is trained, you can use the demo script to try it out:

```bash
# Run with pre-defined examples
python -m src.sentiment_demo

# Run in interactive mode to type your own text
python -m src.sentiment_demo --interactive
```

The demo provides:

- Color-coded sentiment predictions (green for positive, red for negative)
- Confidence scores for each prediction
- Performance metrics for batch processing

### Step 5: Deploy to Hugging Face Hub

Once you're satisfied with your model, you can share it with the community by deploying it to the Hugging Face Hub:

```bash
# Make sure you're logged in to Hugging Face
huggingface-cli login

# Deploy your model (replace YOUR_USERNAME with your Hugging Face username)
python -m src.model.sentiment_publish --repo_name="YOUR_USERNAME/imdb-sentiment-analysis"

# Alternatively, use the Makefile target
make publish-sentiment REPO_NAME="YOUR_USERNAME/imdb-sentiment-analysis"
```

This will:
- Upload your model and tokenizer to Hugging Face Hub
- Create a model card with usage information
- Make your model publicly accessible via the Hugging Face API

After deployment, your model will be available at `https://huggingface.co/YOUR_USERNAME/imdb-sentiment-analysis` and can be used with the transformers library as shown in the demo section above.

#### Customizing Your Model Card

The project includes a template model card (`model_card.md`) that you can customize before deployment. This provides a more detailed presentation of your model on Hugging Face Hub.

To customize and upload your model card:

1. Edit the `model_card.md` file with your information:
   - Update the developer information
   - Fill in your performance metrics
   - Modify example use cases

2. After deploying your model, update the model card on Hugging Face Hub:

```bash
# Using the Python script directly
python -m src.model.update_model_card --repo_name="YOUR_USERNAME/imdb-sentiment-analysis" --model_card="model_card.md"

# Or using the Makefile target
make update-model-card REPO_NAME="YOUR_USERNAME/imdb-sentiment-analysis"
```

This will replace the default README.md on your Hugging Face model page with your customized model card, providing users with comprehensive information about your model.

## Project Structure for Sentiment Analysis

- `config/sentiment_analysis.yaml`: Configuration file
- `src/model/sentiment_train.py`: Training script
- `src/model/sentiment_inference.py`: Inference code
- `src/model/sentiment_evaluate.py`: Evaluation code
- `src/model/sentiment_publish.py`: Hugging Face deployment script
- `src/model/update_model_card.py`: Model card updater for Hugging Face
- `src/sentiment_demo.py`: Interactive demo
- `tests/test_sentiment_model.py`: Test suite
- `model_card.md`: Template for creating detailed model cards

## Customization

You can modify the `config/sentiment_analysis.yaml` file to change:

- Model parameters
- Training settings
- Performance thresholds
- Dataset options

## Next Steps

Once you're comfortable with the sentiment analysis project, consider:

1. **Adding your own test cases** - Create challenging examples to test model robustness
2. **Training with challenging cases** - Improve model performance on difficult examples
3. **Experimenting with different models** - Try other small models like Phi-2 or Gemma 2B
4. **Implementing advanced testing** - Test for bias or concept drift

### Training with Challenging Cases

The project includes a framework for improving model performance on difficult examples:

1. **Identify challenging cases** using the challenge test framework:

```bash
python -m src.sentiment_demo --challenge
```

2. **Add new challenging cases** when you find examples the model struggles with:

```bash
python -m src.utils.add_challenge_case --text "Your text here" --sentiment 0 --category "subtle_negative"
```

3. **Train a new model** using the challenging cases to augment the training data:

```bash
python -m src.model.train_with_challenging_cases --save_dir="models/sentiment_improved"
```

This will:
- Create an augmented dataset that includes your challenging examples
- Train a new model with this enhanced dataset
- Save the model to the specified directory

4. **Evaluate the improved model** on the challenging test set:

```bash
python -m src.model.sentiment_challenge_test --model_dir="models/sentiment_improved" --plot
```

This training approach helps the model learn from its mistakes and improve on the specific types of examples it previously struggled with.

## Testing Philosophy and Methodology

Testing machine learning models is fundamentally different from testing traditional software:

1. **Non-deterministic outputs**: ML models may produce slightly different results even with the same inputs
2. **Performance vs correctness**: We test for acceptable performance rather than 100% correctness
3. **Robustness testing**: We need to test how models handle edge cases and adversarial inputs
4. **Concept drift**: Models can degrade over time as real-world data changes
5. **Data quality impacts**: Testing must account for data biases and quality issues

### Test Suite Overview

Our test suite consists of several types of tests:

#### 1. Unit Tests (`test_model_loading.py`)

Tests for properly loading models and tokenizers:

- Tests that the model loading function works correctly
- Verifies model architecture and configuration
- Ensures tokenizers are properly initialized

#### 2. Functional Tests (`test_sentiment_model.py`)

Tests for model behavior and predictions:

- Verifies the model predicts expected sentiments on obvious examples
- Tests model confidence scoring
- Tests handling of edge cases (empty strings, very long text)
- Tests batch processing capabilities

#### 3. Performance Tests (`test_sentiment_model.py`)

Tests for model performance metrics:

- Verifies accuracy meets minimum thresholds
- Tests inference speed requirements
- Measures memory usage during inference

#### 4. Challenging Test Cases (`sentiment_challenge_test.py`)

Tests for model performance on particularly difficult examples:

- Identifies categories of challenging inputs (subtle negative, sarcasm, etc.)
- Tracks performance across different challenge categories
- Provides a feedback loop for targeted model improvement
- Generates visualizations and reports of model weaknesses

### Running Tests

#### Basic Test Commands

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_sentiment_model.py

# Run a specific test class
pytest tests/test_sentiment_model.py::TestModelPrediction

# Run a specific test method
pytest tests/test_sentiment_model.py::TestModelPrediction::test_single_prediction
```

### Running Challenging Test Cases

The challenging test cases framework provides specialized testing for difficult examples:

```bash
# Run challenge tests through the demo script
python -m src.sentiment_demo --challenge

# Run with detailed analysis and visualization
python -m src.model.sentiment_challenge_test --plot

# Add a new challenging case
python -m src.utils.add_challenge_case --text "Your text here" --sentiment 0 --category "subtle_negative" --notes "Optional notes"

# List available challenge categories
python -m src.utils.add_challenge_case --list-categories
```

The challenging test framework helps you:

1. **Identify model weaknesses** in specific categories of text
2. **Track improvement** as you refine your model
3. **Build a comprehensive test set** of difficult examples
4. **Generate reports and visualizations** of model performance

#### Code Coverage

We track test coverage to ensure our test suite adequately tests the entire codebase:

```bash
# Generate coverage report
pytest --cov=src

# Generate HTML coverage report (output to htmlcov/ directory)
pytest --cov=src --cov-report=html

# Open the coverage report in your browser
open htmlcov/index.html  # On macOS
xdg-open htmlcov/index.html  # On Linux
```

Coverage reports help identify untested or undertested parts of the codebase that may require additional testing.

### Automated Testing Infrastructure

The project includes automated testing infrastructure to ensure code quality:

#### Local Testing

Run all tests and generate coverage reports with a single command:

```bash
./run_tests.sh
```

This script:
- Runs unit tests
- Generates code coverage reports
- Runs challenge tests
- Provides a summary of test results

#### Pre-commit Hooks

Set up git hooks to run tests before each commit:

```bash
./setup_hooks.sh
```

This ensures that only code that passes tests can be committed.

#### Continuous Integration

The project uses GitHub Actions for continuous integration:

- Tests run automatically on each push and pull request
- Code coverage reports are generated and uploaded
- Challenge test results are archived as artifacts

You can view the CI workflow configuration in `.github/workflows/test.yml`.