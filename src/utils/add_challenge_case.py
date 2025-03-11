import os
import json
import argparse
from colorama import Fore, Style, init

# Initialize colorama
init()

def load_challenge_file(file_path):
    """Load the challenge test cases file"""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        # Create a new file with basic structure
        return {
            "version": "1.0",
            "description": "Challenging sentiment analysis test cases",
            "categories": {
                "subtle_negative": {
                    "description": "Negative sentiment expressed with sophisticated vocabulary or subtle phrasing"
                },
                "mixed_sentiment": {
                    "description": "Text containing both positive and negative elements, but with an overall sentiment that should be clear"
                },
                "sarcasm": {
                    "description": "Sarcastic text where the literal meaning differs from the intended sentiment"
                },
                "negation": {
                    "description": "Text using negation that can confuse sentiment models"
                }
            },
            "test_cases": []
        }

def save_challenge_file(data, file_path):
    """Save the challenge test cases file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def add_challenge_case(file_path, text, expected_sentiment, category, notes=None):
    """Add a new challenging case to the file"""
    # Load existing data
    data = load_challenge_file(file_path)
    
    # Check if category exists, add if not
    if category not in data["categories"]:
        print(f"{Fore.YELLOW}Warning: Category '{category}' does not exist. Adding it.{Style.RESET_ALL}")
        data["categories"][category] = {
            "description": f"New category: {category}"
        }
    
    # Check if this text already exists
    for case in data["test_cases"]:
        if case["text"] == text:
            print(f"{Fore.RED}Error: This text already exists in the test cases.{Style.RESET_ALL}")
            return False
    
    # Add new case
    new_case = {
        "text": text,
        "expected_sentiment": expected_sentiment,
        "category": category
    }
    
    if notes:
        new_case["notes"] = notes
    
    data["test_cases"].append(new_case)
    
    # Save updated data
    save_challenge_file(data, file_path)
    print(f"{Fore.GREEN}Successfully added new challenging case.{Style.RESET_ALL}")
    return True

def list_categories(file_path):
    """List all available categories"""
    data = load_challenge_file(file_path)
    print(f"\n{Fore.CYAN}Available Categories:{Style.RESET_ALL}")
    for category, info in data["categories"].items():
        print(f"- {category}: {info.get('description', 'No description')}")

def main():
    parser = argparse.ArgumentParser(description="Add a new challenging test case")
    parser.add_argument("--file", type=str, default="tests/challenging_cases.json",
                        help="Path to challenging test cases file")
    parser.add_argument("--text", type=str, help="The text to add as a challenging case")
    parser.add_argument("--sentiment", type=int, choices=[0, 1], 
                        help="Expected sentiment (0=negative, 1=positive)")
    parser.add_argument("--category", type=str, help="Category of the challenging case")
    parser.add_argument("--notes", type=str, help="Additional notes about this case")
    parser.add_argument("--list-categories", action="store_true", 
                        help="List all available categories")
    args = parser.parse_args()
    
    if args.list_categories:
        list_categories(args.file)
        return
    
    if not all([args.text, args.sentiment is not None, args.category]):
        parser.print_help()
        print(f"\n{Fore.RED}Error: --text, --sentiment, and --category are required.{Style.RESET_ALL}")
        print(f"Use --list-categories to see available categories.")
        return
    
    add_challenge_case(args.file, args.text, args.sentiment, args.category, args.notes)

if __name__ == "__main__":
    main() 