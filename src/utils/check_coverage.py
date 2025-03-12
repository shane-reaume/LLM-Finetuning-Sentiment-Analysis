#!/usr/bin/env python
"""
Script to check if all Python files are being included in the coverage report.
"""

import os
import glob
import argparse
from pathlib import Path


def find_python_files(src_dir):
    """Find all Python files in the source directory."""
    python_files = []
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files


def find_covered_files(coverage_dir):
    """Find all Python files that have coverage data."""
    covered_files = []
    html_files = glob.glob(os.path.join(coverage_dir, '**/*.html'), recursive=True)
    
    for html_file in html_files:
        # Extract the Python file path from the HTML file name
        # HTML files are named like: src_model_sentiment_train_py.html
        file_name = os.path.basename(html_file)
        if file_name == 'index.html':
            continue
            
        # Convert HTML file name back to Python file path
        parts = file_name.replace('.html', '').split('_')
        if len(parts) < 2:
            continue
            
        # Reconstruct the Python file path
        py_path = os.path.join(*parts[:-1]) + '.py'
        covered_files.append(py_path)
        
    return covered_files


def main():
    parser = argparse.ArgumentParser(description="Check coverage of Python files")
    parser.add_argument(
        "--src-dir",
        default="src",
        help="Source directory containing Python files"
    )
    parser.add_argument(
        "--coverage-dir",
        default="htmlcov",
        help="Directory containing coverage HTML files"
    )
    
    args = parser.parse_args()
    
    # Find all Python files
    python_files = find_python_files(args.src_dir)
    print(f"Found {len(python_files)} Python files in {args.src_dir}")
    
    # Find all covered files
    covered_files = find_covered_files(args.coverage_dir)
    print(f"Found {len(covered_files)} covered files in {args.coverage_dir}")
    
    # Find files that are not covered
    not_covered = []
    for py_file in python_files:
        rel_path = os.path.relpath(py_file)
        if not any(rel_path.endswith(covered) for covered in covered_files):
            not_covered.append(rel_path)
    
    # Print results
    if not_covered:
        print("\nPython files not included in coverage report:")
        for file in sorted(not_covered):
            print(f"  - {file}")
    else:
        print("\nAll Python files are included in the coverage report!")
    
    # Print coverage percentage
    if python_files:
        coverage_pct = (len(python_files) - len(not_covered)) / len(python_files) * 100
        print(f"\nCoverage: {coverage_pct:.1f}% ({len(python_files) - len(not_covered)}/{len(python_files)} files)")


if __name__ == "__main__":
    main() 