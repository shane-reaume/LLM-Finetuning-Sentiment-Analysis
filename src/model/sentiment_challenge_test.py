import os
import json
import argparse
import time
from tabulate import tabulate
from colorama import Fore, Style, init
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from src.model.sentiment_inference import SentimentClassifier
from src.utils.config_utils import load_config

# Initialize colorama
init()

def load_challenging_cases(file_path):
    """Load challenging test cases from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def colorize_sentiment(text, prediction, expected, confidence):
    """Apply color to text based on prediction correctness"""
    sentiment_text = "POSITIVE" if prediction == 1 else "NEGATIVE"
    
    if prediction == expected:
        # Correct prediction
        return f"{Fore.GREEN}[{sentiment_text} {confidence:.2f}] ✓{Style.RESET_ALL} {text}"
    else:
        # Incorrect prediction
        return f"{Fore.RED}[{sentiment_text} {confidence:.2f}] ✗{Style.RESET_ALL} {text}"

def run_challenge_tests(classifier, test_cases, verbose=True):
    """Run tests on challenging cases and return results"""
    results = []
    
    for case in test_cases:
        text = case["text"]
        expected = case["expected_sentiment"]
        category = case["category"]
        notes = case.get("notes", "")
        
        # Get prediction
        start_time = time.time()
        result = classifier.predict(text, return_confidence=True)
        
        # Handle both tuple and single value returns
        if isinstance(result, tuple) and len(result) == 3:
            prediction, confidence, inference_time = result
        else:
            prediction = result
            confidence = 1.0
            inference_time = time.time() - start_time
        
        # Store result
        correct = prediction == expected
        results.append({
            "text": text,
            "expected": expected,
            "prediction": prediction,
            "confidence": confidence,
            "correct": correct,
            "category": category,
            "notes": notes,
            "inference_time": inference_time
        })
        
        # Print result if verbose
        if verbose:
            print(colorize_sentiment(text, prediction, expected, confidence))
            print(f"Category: {category}")
            if notes:
                print(f"Notes: {notes}")
            print(f"Inference time: {inference_time * 1000:.2f} ms")
            print("-" * 80)
    
    return results

def print_summary(results):
    """Print summary of test results"""
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n===== CHALLENGE TEST SUMMARY =====")
    print(f"Total test cases: {total}")
    print(f"Correctly classified: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Category breakdown
    print("\n===== CATEGORY BREAKDOWN =====")
    categories = defaultdict(lambda: {"total": 0, "correct": 0})
    
    for r in results:
        cat = r["category"]
        categories[cat]["total"] += 1
        if r["correct"]:
            categories[cat]["correct"] += 1
    
    table_data = []
    for cat, stats in categories.items():
        cat_accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        table_data.append([
            cat, 
            stats["total"], 
            stats["correct"], 
            f"{cat_accuracy:.2%}"
        ])
    
    print(tabulate(
        table_data, 
        headers=["Category", "Total", "Correct", "Accuracy"],
        tablefmt="grid"
    ))
    
    return categories

def plot_category_performance(categories):
    """Plot performance by category"""
    cats = list(categories.keys())
    accuracies = [categories[c]["correct"] / categories[c]["total"] 
                 if categories[c]["total"] > 0 else 0 for c in cats]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(cats, accuracies, color='skyblue')
    
    # Add value labels on top of bars
    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            acc + 0.02,
            f'{acc:.2%}',
            ha='center',
            va='bottom'
        )
    
    plt.ylim(0, 1.1)  # Set y-axis limit
    plt.xlabel('Category')
    plt.ylabel('Accuracy')
    plt.title('Model Performance by Challenge Category')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save plot
    os.makedirs('reports', exist_ok=True)
    plt.savefig('reports/challenge_category_performance.png')
    print("Category performance plot saved to reports/challenge_category_performance.png")

def save_results(results, output_file):
    """Save test results to JSON file"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results,
            "summary": {
                "total": len(results),
                "correct": sum(1 for r in results if r["correct"]),
                "accuracy": sum(1 for r in results if r["correct"]) / len(results) if results else 0
            }
        }, f, indent=2)
    
    print(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Run sentiment analysis on challenging test cases")
    parser.add_argument("--config", type=str, default="config/sentiment_analysis.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Path to model directory (overrides config)")
    parser.add_argument("--test_file", type=str, default="tests/challenging_cases.json",
                        help="Path to challenging test cases file")
    parser.add_argument("--output", type=str, default="reports/challenge_test_results.json",
                        help="Path to output results file")
    parser.add_argument("--plot", action="store_true", help="Generate performance plots")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Use provided model_dir or default from config
    model_path = args.model_dir if args.model_dir else config["model"]["save_dir"]
    
    print(f"Loading sentiment analysis model from {model_path}...")
    classifier = SentimentClassifier(model_path)
    print("Model loaded successfully!")
    
    # Load challenging cases
    print(f"Loading challenging test cases from {args.test_file}...")
    test_data = load_challenging_cases(args.test_file)
    test_cases = test_data["test_cases"]
    print(f"Loaded {len(test_cases)} challenging test cases")
    
    # Run tests
    print("\n===== RUNNING CHALLENGE TESTS =====\n")
    results = run_challenge_tests(classifier, test_cases, verbose=not args.quiet)
    
    # Print summary
    categories = print_summary(results)
    
    # Generate plot if requested
    if args.plot:
        plot_category_performance(categories)
    
    # Save results
    save_results(results, args.output)

if __name__ == "__main__":
    main() 