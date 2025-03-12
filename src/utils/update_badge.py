#!/usr/bin/env python
"""
Script to manually update the challenge tests badge on GitHub Pages.
This can be used if the automatic badge generation in GitHub Actions is not working.
"""

import os
import json
import argparse
import subprocess
from pathlib import Path


def update_badge(accuracy_pct, output_dir="gh-pages"):
    """
    Update the challenge tests badge with a specific accuracy percentage.
    
    Args:
        accuracy_pct: Accuracy percentage (0-100)
        output_dir: Directory to save the badge JSON file
    """
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
        
    # Create badge data
    badge_data = {
        "schemaVersion": 1,
        "label": "challenge tests",
        "message": f"{accuracy_pct}%",
        "color": color
    }
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Write badge data to file
    output_file = Path(output_dir) / "challenge-tests-badge.json"
    with open(output_file, 'w') as f:
        json.dump(badge_data, f, indent=2)
        
    print(f"Badge updated at {output_file}")
    
    # Check if we're in a git repository
    try:
        subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], 
                      check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Check if gh-pages branch exists
        result = subprocess.run(["git", "branch", "-a"], 
                               check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        branches = result.stdout.decode('utf-8')
        
        if "gh-pages" in branches or "origin/gh-pages" in branches:
            print("Found gh-pages branch. You can push the badge with:")
            print(f"git add {output_file}")
            print("git commit -m 'Update challenge tests badge'")
            print("git push origin gh-pages")
    except subprocess.CalledProcessError:
        # Not in a git repository or git not installed
        pass


def main():
    parser = argparse.ArgumentParser(description="Update challenge tests badge")
    parser.add_argument(
        "--accuracy",
        type=int,
        required=True,
        help="Accuracy percentage (0-100)"
    )
    parser.add_argument(
        "--output-dir",
        default="gh-pages",
        help="Directory to save badge JSON file"
    )
    
    args = parser.parse_args()
    update_badge(args.accuracy, args.output_dir)


if __name__ == "__main__":
    main() 