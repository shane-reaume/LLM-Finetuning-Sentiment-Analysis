#!/usr/bin/env python
"""
Script to generate dynamic badges for GitHub README based on test results.
This script reads the challenge test results and generates a badge JSON file
that can be used with shields.io's endpoint badge.
"""

import json
import os
import argparse
from pathlib import Path


def generate_challenge_test_badge(results_file, output_dir):
    """
    Generate a badge JSON file for challenge tests based on test results.
    
    Args:
        results_file: Path to the challenge test results JSON file
        output_dir: Directory to save the badge JSON file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Default badge data (used if we can't read the results)
    badge_data = {
        "schemaVersion": 1,
        "label": "challenge tests",
        "message": "unknown",
        "color": "lightgrey"
    }
    
    try:
        # Check if results file exists
        if not os.path.exists(results_file):
            print(f"Warning: Results file {results_file} not found. Using default badge.")
            badge_data["message"] = "no data"
        else:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # Extract overall accuracy from results
            if 'summary' in results and 'accuracy' in results['summary']:
                accuracy = results['summary']['accuracy']
            elif 'overall_accuracy' in results:
                accuracy = results['overall_accuracy']
            elif 'accuracy' in results:
                accuracy = results['accuracy']
            else:
                # Calculate from individual results if available
                correct = sum(1 for item in results.get('results', []) if item.get('correct', False))
                total = len(results.get('results', []))
                accuracy = correct / total if total > 0 else 0
                
            # Format accuracy as percentage
            accuracy_pct = round(accuracy * 100)
            
            # Determine color based on accuracy
            if accuracy_pct >= 90:
                color = "brightgreen"
            elif accuracy_pct >= 80:
                color = "green"
            elif accuracy_pct >= 70:
                color = "yellowgreen"
            elif accuracy_pct >= 60:
                color = "yellow"
            else:
                color = "red"
                
            # Update badge data
            badge_data["message"] = f"{accuracy_pct}%"
            badge_data["color"] = color
        
        # Write badge data to file
        output_file = Path(output_dir) / "challenge-tests-badge.json"
        with open(output_file, 'w') as f:
            json.dump(badge_data, f, indent=2)
            
        print(f"Badge generated at {output_file}")
        
    except Exception as e:
        print(f"Error generating badge: {e}")
        # Write the default badge data to file
        output_file = Path(output_dir) / "challenge-tests-badge.json"
        with open(output_file, 'w') as f:
            json.dump(badge_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate badges for GitHub README")
    parser.add_argument(
        "--results-file",
        default="reports/challenge_test_results.json",
        help="Path to challenge test results JSON file"
    )
    parser.add_argument(
        "--output-dir",
        default="gh-pages",
        help="Directory to save badge JSON files"
    )
    
    args = parser.parse_args()
    generate_challenge_test_badge(args.results_file, args.output_dir)


if __name__ == "__main__":
    main() 