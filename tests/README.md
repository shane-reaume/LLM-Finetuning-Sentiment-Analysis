# Sentiment Analysis Testing

This directory contains testing tools for the sentiment analysis model, including specialized tests for challenging cases.

## Challenging Test Cases

The `challenging_cases.json` file contains examples that are particularly difficult for sentiment analysis models to classify correctly. These include:

- **Subtle negative expressions**: Negative sentiment expressed with sophisticated vocabulary
- **Mixed sentiment**: Text with both positive and negative elements
- **Sarcasm**: Text where the literal meaning differs from the intended sentiment
- **Negation**: Text using negation that can confuse sentiment models

## Running Challenge Tests

You can run the challenge tests in several ways:

### 1. Using the sentiment_demo.py script

```bash
python -m src.sentiment_demo --challenge
```

This will run all the challenging test cases and display a summary of the results.

### 2. Using the dedicated challenge test script

For more detailed analysis and visualization:

```bash
python -m src.model.sentiment_challenge_test --plot
```

Options:
- `--plot`: Generate performance plots
- `--quiet`: Suppress detailed output
- `--output`: Specify output file path
- `--test_file`: Specify a different test file

### 3. Adding new challenging cases

When you find examples that your model misclassifies, you can add them to the test set:

```bash
python -m src.utils.add_challenge_case --text "Your text here" --sentiment 0 --category "subtle_negative" --notes "Optional notes"
```

To see available categories:

```bash
python -m src.utils.add_challenge_case --list-categories
```

## Using Challenge Tests for Model Improvement

The challenge test system provides a feedback loop for improving your model:

1. **Identify weaknesses**: Run challenge tests to find categories where your model struggles
2. **Add examples**: When you find misclassifications in real usage, add them to the test set
3. **Targeted training**: Use the challenge categories to guide additional training data collection
4. **Track progress**: Compare model versions using the same challenge test set

## Interpreting Results

The test results show:
- Overall accuracy on challenging cases
- Breakdown by category
- Individual examples with predictions and confidence scores

Low performance in specific categories indicates areas where your model needs improvement. 