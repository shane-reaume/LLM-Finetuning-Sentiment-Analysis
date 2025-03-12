#!/usr/bin/env python
"""
Script to identify files with low test coverage.
This script parses the coverage.xml file and outputs a list of files
with coverage below a specified threshold, sorted by coverage percentage.
"""

import os
import sys
import argparse
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional
import json


def parse_coverage_xml(coverage_file: str) -> List[Tuple[str, float, int, int]]:
    """
    Parse the coverage.xml file and extract coverage information for each file.
    
    Args:
        coverage_file: Path to the coverage.xml file
        
    Returns:
        List of tuples containing (filename, coverage_percentage, covered_lines, total_lines)
    """
    if not os.path.exists(coverage_file):
        print(f"Error: Coverage file {coverage_file} not found.")
        print("Run tests with coverage first: ./run_tests_with_coverage.sh")
        sys.exit(1)
    
    try:
        tree = ET.parse(coverage_file)
        root = tree.getroot()
        
        # Extract coverage data for each file
        coverage_data = []
        
        for package in root.findall('.//package'):
            for class_elem in package.findall('.//class'):
                filename = class_elem.attrib.get('filename', '')
                
                # Skip if not a Python file or in tests directory
                if not filename.endswith('.py') or filename.startswith('tests/'):
                    continue
                
                # Get line coverage data
                line_rate = float(class_elem.attrib.get('line-rate', 0))
                coverage_percentage = line_rate * 100
                
                # Count covered and total lines
                lines = class_elem.findall('.//line')
                total_lines = len(lines)
                covered_lines = sum(1 for line in lines if line.attrib.get('hits', '0') != '0')
                
                coverage_data.append((filename, coverage_percentage, covered_lines, total_lines))
        
        return coverage_data
    
    except Exception as e:
        print(f"Error parsing coverage file: {e}")
        sys.exit(1)


def generate_report(coverage_data: List[Tuple[str, float, int, int]], threshold: float, output_format: str, output_file: Optional[str] = None) -> None:
    """
    Generate a report of files with coverage below the threshold.
    
    Args:
        coverage_data: List of tuples containing (filename, coverage_percentage, covered_lines, total_lines)
        threshold: Coverage threshold percentage
        output_format: Format of the output ('text', 'json', or 'markdown')
        output_file: Path to the output file (optional)
    """
    # Sort by coverage percentage (ascending)
    sorted_data = sorted(coverage_data, key=lambda x: x[1])
    
    # Filter files below threshold
    low_coverage_files = [item for item in sorted_data if item[1] < threshold]
    
    # Calculate overall statistics
    total_files = len(coverage_data)
    low_coverage_count = len(low_coverage_files)
    average_coverage = sum(item[1] for item in coverage_data) / total_files if total_files > 0 else 0
    
    # Generate report based on format
    if output_format == 'json':
        report = {
            "summary": {
                "total_files": total_files,
                "low_coverage_files": low_coverage_count,
                "average_coverage": average_coverage,
                "threshold": threshold
            },
            "low_coverage_files": [
                {
                    "filename": item[0],
                    "coverage": item[1],
                    "covered_lines": item[2],
                    "total_lines": item[3]
                }
                for item in low_coverage_files
            ]
        }
        
        output = json.dumps(report, indent=2)
    
    elif output_format == 'markdown':
        output = f"# Test Coverage Report\n\n"
        output += f"## Summary\n\n"
        output += f"- **Total Files**: {total_files}\n"
        output += f"- **Files Below Threshold ({threshold}%)**: {low_coverage_count}\n"
        output += f"- **Average Coverage**: {average_coverage:.2f}%\n\n"
        
        output += f"## Files with Coverage Below {threshold}%\n\n"
        output += "| File | Coverage | Covered Lines | Total Lines |\n"
        output += "|------|----------|--------------|-------------|\n"
        
        for filename, coverage, covered, total in low_coverage_files:
            output += f"| {filename} | {coverage:.2f}% | {covered} | {total} |\n"
        
        # Add recommendations
        output += "\n## Recommendations\n\n"
        
        if low_coverage_files:
            output += "Focus on improving test coverage for these files, starting with the most critical ones:\n\n"
            for i, (filename, coverage, covered, total) in enumerate(low_coverage_files[:5], 1):
                output += f"{i}. `{filename}` ({coverage:.2f}%): Add tests for the main functionality\n"
        else:
            output += "All files meet the minimum coverage threshold. Great job!\n"
    
    else:  # text format
        output = f"Test Coverage Report\n"
        output += f"===================\n\n"
        output += f"Summary:\n"
        output += f"  Total Files: {total_files}\n"
        output += f"  Files Below Threshold ({threshold}%)**: {low_coverage_count}\n"
        output += f"  Average Coverage: {average_coverage:.2f}%\n\n"
        
        output += f"Files with Coverage Below {threshold}%:\n"
        output += f"------------------------------------\n"
        
        for filename, coverage, covered, total in low_coverage_files:
            output += f"  {filename}: {coverage:.2f}% ({covered}/{total} lines)\n"
        
        # Add recommendations
        output += "\nRecommendations:\n"
        output += "---------------\n"
        
        if low_coverage_files:
            output += "Focus on improving test coverage for these files, starting with the most critical ones:\n"
            for i, (filename, coverage, covered, total) in enumerate(low_coverage_files[:5], 1):
                output += f"{i}. {filename} ({coverage:.2f}%): Add tests for the main functionality\n"
        else:
            output += "All files meet the minimum coverage threshold. Great job!\n"
    
    # Output the report
    if output_file:
        with open(output_file, 'w') as f:
            f.write(output)
        print(f"Report written to {output_file}")
    else:
        print(output)


def main():
    """Main function to parse arguments and generate the report."""
    parser = argparse.ArgumentParser(description='Identify files with low test coverage.')
    parser.add_argument('--coverage-file', default='coverage.xml',
                        help='Path to the coverage.xml file (default: coverage.xml)')
    parser.add_argument('--threshold', type=float, default=70.0,
                        help='Coverage threshold percentage (default: 70.0)')
    parser.add_argument('--format', choices=['text', 'json', 'markdown'], default='text',
                        help='Output format (default: text)')
    parser.add_argument('--output', help='Output file (default: stdout)')
    
    args = parser.parse_args()
    
    # Parse coverage data
    coverage_data = parse_coverage_xml(args.coverage_file)
    
    # Generate and output the report
    generate_report(coverage_data, args.threshold, args.format, args.output)


if __name__ == '__main__':
    main() 