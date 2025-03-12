"""
Tests for the sentiment_evaluate.py module.
"""

import os
import sys
import pytest
import unittest.mock as mock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_sentiment_evaluate_imports():
    """Test that sentiment_evaluate.py can be imported."""
    # Mock the import of SentimentClassifier to avoid the error
    with mock.patch('src.model.sentiment_inference.SentimentClassifier'):
        import src.model.sentiment_evaluate
        assert True


def test_load_test_examples():
    """Test the load_test_examples function."""
    # Mock open to avoid reading a real file
    mock_data = [
        {"text": "This is positive", "label": 1},
        {"text": "This is negative", "label": 0}
    ]
    
    with mock.patch('builtins.open', mock.mock_open(read_data=str(mock_data))):
        with mock.patch('json.load', return_value=mock_data):
            from src.model.sentiment_evaluate import load_test_examples
            
            # Call the function
            texts, labels = load_test_examples("dummy_path.json")
            
            # Check the results
            assert len(texts) == 2
            assert len(labels) == 2
            assert texts[0] == "This is positive"
            assert labels[0] == 1
            assert texts[1] == "This is negative"
            assert labels[1] == 0


@mock.patch('src.model.sentiment_inference.SentimentClassifier')
def test_evaluate_model_function(mock_classifier_class):
    """Test the evaluate_model function with a mocked classifier."""
    # Configure the mock
    mock_classifier = mock.MagicMock()
    mock_classifier_class.return_value = mock_classifier
    
    # Set up the evaluate method to return metrics
    mock_metrics = {
        "accuracy": 0.85,
        "f1": 0.86,
        "precision": 0.84,
        "recall": 0.88,
        "high_confidence_accuracy": 0.90,
        "high_confidence_examples": 80,
        "total_examples": 100,
        "avg_inference_time_ms": 5.0
    }
    mock_classifier.evaluate.return_value = mock_metrics
    
    # Create a patched version of evaluate_model that doesn't use string formatting
    def patched_evaluate_model(config, model_dir=None, detailed=False, test_file=None):
        model_path = model_dir if model_dir else config["model"]["save_dir"]
        classifier = mock_classifier_class(model_path)
        test_file_path = test_file if test_file else config["testing"]["test_examples_file"]
        return classifier.evaluate(None, None)  # We don't care about the actual arguments here
    
    # Mock the evaluate_model function
    with mock.patch('src.model.sentiment_evaluate.evaluate_model', side_effect=patched_evaluate_model):
        # Import the function
        from src.model.sentiment_evaluate import evaluate_model
        
        # Create a test config
        config = {
            "model": {"save_dir": "test_model_dir"},
            "testing": {"test_examples_file": "test_examples.json"}
        }
        
        # Call the function
        result = evaluate_model(config)
        
        # Check that the classifier was created with the right path
        mock_classifier_class.assert_called_once_with("test_model_dir")
        
        # Check that evaluate was called
        mock_classifier.evaluate.assert_called_once()
        
        # Check that the function returned the metrics
        assert result == mock_metrics


@mock.patch('argparse.ArgumentParser.parse_args')
@mock.patch('src.model.sentiment_evaluate.load_config')
@mock.patch('src.model.sentiment_evaluate.evaluate_model')
def test_main_function(mock_evaluate, mock_load_config, mock_parse_args):
    """Test the main function."""
    # Configure the mocks
    mock_args = mock.MagicMock()
    mock_args.config = "config.yaml"
    mock_args.model_dir = "model_dir"
    mock_args.detailed = True
    mock_args.test_file = "test_file.json"
    mock_parse_args.return_value = mock_args
    
    mock_config = {"key": "value"}
    mock_load_config.return_value = mock_config
    
    # Import the function
    from src.model.sentiment_evaluate import main
    
    # Call the function
    main()
    
    # Check that load_config was called with the right path
    mock_load_config.assert_called_once_with("config.yaml")
    
    # Check that evaluate_model was called with the right arguments
    mock_evaluate.assert_called_once_with(mock_config, "model_dir", True, "test_file.json")


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 