"""
Tests for the sentiment_create_test_set.py module.
"""

import os
import sys
import json
import pytest
import tempfile
import unittest.mock as mock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_sentiment_create_test_set_imports():
    """Test that sentiment_create_test_set.py can be imported."""
    import src.data.sentiment_create_test_set
    assert True


@mock.patch('json.dump')
@mock.patch('builtins.open', new_callable=mock.mock_open)
def test_save_test_examples(mock_open, mock_json_dump):
    """Test saving test examples to a file."""
    # Import the main function which contains the save operation
    from src.data.sentiment_create_test_set import main
    
    # Mock the create_test_set function to return test data
    with mock.patch('src.data.sentiment_create_test_set.create_test_set') as mock_create:
        # Create test data
        test_data = [
            {"text": "This is a positive review", "label": 1},
            {"text": "This is a negative review", "label": 0}
        ]
        mock_create.return_value = test_data
        
        # Mock the load_config function
        with mock.patch('src.data.sentiment_create_test_set.load_config') as mock_load_config:
            mock_load_config.return_value = {
                "data": {
                    "test_examples_dir": "data/test_examples"
                }
            }
            
            # Mock os.makedirs to avoid directory creation issues
            with mock.patch('os.makedirs') as mock_makedirs:
                # Mock the argparse.ArgumentParser
                with mock.patch('argparse.ArgumentParser.parse_args') as mock_args:
                    mock_args.return_value = mock.MagicMock(
                        config="config/sentiment_analysis.yaml",
                        num_examples=2,
                        output="test_output.json"
                    )
                    
                    # Call the main function
                    main()
                    
                    # Check that the file was opened correctly
                    mock_open.assert_called_once_with('test_output.json', 'w')
                    
                    # Check that json.dump was called with the test data
                    mock_json_dump.assert_called_once()
                    args, _ = mock_json_dump.call_args
                    assert args[0] == test_data


@mock.patch('datasets.load_dataset')
def test_create_test_set(mock_load_dataset):
    """Test the create_test_set function."""
    from src.data.sentiment_create_test_set import create_test_set
    
    # Create a mock dataset
    mock_dataset = mock.MagicMock()
    mock_test_split = mock.MagicMock()
    
    # Set up the test data
    test_data = [
        {"text": "This is a positive review", "label": 1},
        {"text": "This is another positive review", "label": 1},
        {"text": "This is a negative review", "label": 0},
        {"text": "This is another negative review", "label": 0}
    ]
    
    # Configure the mock dataset
    mock_test_split.__iter__.return_value = test_data
    mock_test_split.select.return_value = test_data
    mock_dataset.__getitem__.return_value = mock_test_split
    mock_load_dataset.return_value = mock_dataset
    
    # Create a mock config
    config = {
        "data": {
            "dataset_name": "imdb",
            "cache_dir": "./data/processed",
            "validation_split": "test"
        }
    }
    
    # Call the function
    result = create_test_set(config, num_examples=4)
    
    # Check the result
    assert len(result) == 4  # Should return 4 test cases
    assert all("text" in item and "label" in item for item in result)  # Each item should have text and label


@mock.patch('os.makedirs')
@mock.patch('json.dump')
@mock.patch('builtins.open', new_callable=mock.mock_open)
@mock.patch('src.data.sentiment_create_test_set.create_test_set')
def test_main_function(mock_create, mock_open, mock_json_dump, mock_makedirs):
    """Test the main function."""
    from src.data.sentiment_create_test_set import main
    
    # Configure the mocks
    mock_create.return_value = [
        {"text": "Test 1", "label": 1},
        {"text": "Test 2", "label": 0}
    ]
    
    # Mock the argparse.ArgumentParser
    with mock.patch('argparse.ArgumentParser.parse_args') as mock_args:
        mock_args.return_value = mock.MagicMock(
            config="config/sentiment_analysis.yaml",
            num_examples=2,
            output="data/processed/test_examples.json"
        )
        
        # Mock the load_config function
        with mock.patch('src.data.sentiment_create_test_set.load_config') as mock_config:
            mock_config.return_value = {
                "testing": {
                    "test_examples_file": "data/processed/test_examples.json"
                }
            }
            
            # Call the function
            main()
            
            # Verify create_test_set was called
            mock_create.assert_called_once()
            
            # Verify json.dump was called with the correct arguments
            mock_json_dump.assert_called_once()
            args, kwargs = mock_json_dump.call_args
            assert args[0] == mock_create.return_value  # First arg should be the test data


def test_integration_with_temp_file():
    """Integration test using a temporary file."""
    # Skip this test for now as it requires actual dataset access
    pytest.skip("Skipping integration test that requires dataset access")


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 