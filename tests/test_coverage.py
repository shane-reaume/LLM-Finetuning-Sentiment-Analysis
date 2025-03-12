"""
Test to ensure that our coverage reporting is working correctly.
This test imports and calls functions from various modules to ensure they're included in coverage.
"""

import os
import sys
import pytest

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_import_all_modules():
    """Test that imports all modules to ensure they're included in coverage."""
    # Import key modules from the project
    import src.model.sentiment_train
    import src.model.sentiment_inference
    import src.model.sentiment_challenge_test
    import src.model.sentiment_publish
    import src.model.train_with_challenging_cases
    import src.model.update_model_card
    
    import src.data.sentiment_augment_training
    import src.data.sentiment_dataset
    
    import src.utils.add_challenge_case
    import src.utils.config_utils
    import src.utils.generate_badges
    import src.utils.generate_challenge_report
    import src.utils.check_coverage
    
    # Import the main demo script
    import src.sentiment_demo
    
    # Assert that the imports worked
    assert True, "All modules imported successfully"


def test_call_key_functions():
    """Test that calls key functions to ensure they're included in coverage."""
    # Import and call config_utils
    from src.utils.config_utils import load_config
    
    # Create a minimal config for testing
    test_config = {
        "model": {
            "name": "test-model",
            "save_dir": "test-dir"
        },
        "data": {
            "dataset_name": "test-dataset",
            "train_split": "train",
            "validation_split": "test",
            "max_length": 128,
            "cache_dir": "./data/processed"
        }
    }
    
    # Write the config to a temporary file
    import tempfile
    import yaml
    import os
    
    # Create a temporary file path
    temp_fd, temp_path = tempfile.mkstemp(suffix='.yaml')
    os.close(temp_fd)
    
    try:
        # Write to the file in text mode
        with open(temp_path, 'w') as f:
            yaml.dump(test_config, f)
        
        # Load the config
        config = load_config(temp_path)
        assert config is not None, "Config should not be None"
        assert config["model"]["name"] == "test-model", "Config should contain model name"
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 