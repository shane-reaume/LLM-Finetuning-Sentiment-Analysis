import os
import sys
import json
import argparse
import time
from colorama import Fore, Style, init

from src.model.sentiment_inference import SentimentClassifier
from src.utils.config_utils import load_config

# Initialize colorama
init()

def colorize_sentiment(text, sentiment, score):
    """Apply color to text based on sentiment"""
    if sentiment == 1:  # Positive
        return f"{Fore.GREEN}[POSITIVE {score:.2f}]{Style.RESET_ALL} {text}"
    else:  # Negative
        return f"{Fore.RED}[NEGATIVE {score:.2f}]{Style.RESET_ALL} {text}"

def load_example_texts():
    """Load some example texts to analyze"""
    return [
        "This movie was absolutely amazing, I loved every minute of it!",
        "The acting was terrible and the plot made no sense at all.",
        "While it had some good moments, overall I was disappointed.",
        "The visual effects were stunning, but the characters felt flat.",
        "I can't recommend this film enough, it's a true masterpiece."
    ]

def main():
    parser = argparse.ArgumentParser(description="Interactive sentiment analysis demo")
    parser.add_argument("--config", type=str, default="config/sentiment_analysis.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Path to model directory (overrides config)")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--challenge", action="store_true",
                        help="Run challenge test cases")
    parser.add_argument("--challenge_file", type=str, default="tests/challenging_cases.json",
                        help="Path to challenging test cases file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Use provided model_dir or default from config
    model_path = args.model_dir if args.model_dir else config["model"]["save_dir"]
    
    print(f"Loading sentiment analysis model from {model_path}...")
    classifier = SentimentClassifier(model_path)
    print("Model loaded successfully!")
    
    if args.challenge:
        # Import challenge test module
        from src.model.sentiment_challenge_test import load_challenging_cases, run_challenge_tests, print_summary
        
        # Load challenging cases
        print(f"Loading challenging test cases from {args.challenge_file}...")
        test_data = load_challenging_cases(args.challenge_file)
        test_cases = test_data["test_cases"]
        print(f"Loaded {len(test_cases)} challenging test cases")
        
        # Run tests
        print("\n===== RUNNING CHALLENGE TESTS =====\n")
        results = run_challenge_tests(classifier, test_cases)
        
        # Print summary
        print_summary(results)
        
    elif args.interactive:
        print("\n===== INTERACTIVE SENTIMENT ANALYSIS DEMO =====")
        print("Type a text and press Enter to analyze its sentiment.")
        print("Type 'quit', 'exit', or Ctrl+C to exit.")
        
        while True:
            try:
                text = input("\nEnter text: ")
                if text.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not text.strip():
                    continue
                
                start_time = time.time()
                result = classifier.predict(text, return_confidence=True)
                total_time = time.time() - start_time
                
                # Handle both tuple and single value returns
                if isinstance(result, tuple) and len(result) == 3:
                    prediction, confidence, inference_time = result
                else:
                    # If only a single value is returned
                    prediction = result
                    confidence = 1.0  # Default confidence
                    inference_time = total_time  # Use total time as inference time
                
                print(colorize_sentiment(text, prediction, confidence))
                print(f"Inference time: {inference_time * 1000:.2f} ms")
                print(f"Total processing time: {total_time * 1000:.2f} ms")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nThank you for using the sentiment analysis demo!")
    
    else:
        # Use example texts
        examples = load_example_texts()
        print("\n===== SENTIMENT ANALYSIS EXAMPLES =====\n")
        
        all_start = time.time()
        for text in examples:
            result = classifier.predict(text, return_confidence=True)
            
            # Handle both tuple and single value returns
            if isinstance(result, tuple) and len(result) == 3:
                prediction, confidence, _ = result
            else:
                # If only a single value is returned
                prediction = result
                confidence = 1.0  # Default confidence
            
            print(colorize_sentiment(text, prediction, confidence))
        
        total_time = time.time() - all_start
        print(f"\nProcessed {len(examples)} examples in {total_time:.2f} seconds "
              f"({total_time/len(examples)*1000:.2f} ms/example)")

if __name__ == "__main__":
    main()