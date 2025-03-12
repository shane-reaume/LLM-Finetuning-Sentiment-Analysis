"""
Tests for the update_badge.py module.
"""

import os
import sys
import json
import pytest
import tempfile
import unittest.mock as mock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_update_badge_imports():
    """Test that update_badge.py can be imported."""
    import src.utils.update_badge
    assert True


def test_update_badge_function():
    """Test the update_badge function."""
    from src.utils.update_badge import update_badge
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Call the function
        update_badge(75, temp_dir)
        
        # Check that the badge file was created
        badge_file = os.path.join(temp_dir, "challenge-tests-badge.json")
        assert os.path.exists(badge_file)
        
        # Check the contents of the badge file
        with open(badge_file, 'r') as f:
            badge_data = json.load(f)
        
        assert badge_data["schemaVersion"] == 1
        assert badge_data["label"] == "challenge tests"
        assert badge_data["message"] == "75%"
        assert badge_data["color"] == "yellowgreen"  # 75% should be yellowgreen


@pytest.mark.parametrize("accuracy,expected_color", [
    (95, "brightgreen"),
    (85, "green"),
    (75, "yellowgreen"),
    (65, "yellow"),
    (55, "red"),
])
def test_badge_colors(accuracy, expected_color):
    """Test that different accuracy values result in the correct badge colors."""
    from src.utils.update_badge import update_badge
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Call the function
        update_badge(accuracy, temp_dir)
        
        # Check the badge color
        badge_file = os.path.join(temp_dir, "challenge-tests-badge.json")
        with open(badge_file, 'r') as f:
            badge_data = json.load(f)
        
        assert badge_data["color"] == expected_color


@mock.patch('subprocess.run')
def test_git_repository_detection(mock_run):
    """Test that git repository detection works."""
    from src.utils.update_badge import update_badge
    
    # Configure the mock to simulate being in a git repository with a gh-pages branch
    mock_run.side_effect = [
        mock.MagicMock(),  # First call to check if in git repo
        mock.MagicMock(stdout=b"* main\n  remotes/origin/gh-pages")  # Second call to list branches
    ]
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Call the function
        update_badge(75, temp_dir)
        
        # Verify the mock was called correctly
        assert mock_run.call_count == 2
        
        # First call should check if in git repo
        args, kwargs = mock_run.call_args_list[0]
        assert args[0][0] == "git"
        assert "rev-parse" in args[0]
        
        # Second call should list branches
        args, kwargs = mock_run.call_args_list[1]
        assert args[0][0] == "git"
        assert "branch" in args[0]


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 