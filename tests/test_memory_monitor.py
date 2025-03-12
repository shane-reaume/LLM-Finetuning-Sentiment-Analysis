"""
Tests for the memory_monitor.py module.
"""

import os
import sys
import time
import pytest
import unittest.mock as mock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_memory_monitor_imports():
    """Test that memory_monitor.py can be imported."""
    import src.utils.memory_monitor
    assert True


@mock.patch('torch.cuda.is_available', return_value=True)
@mock.patch('torch.cuda.memory_allocated')
@mock.patch('torch.cuda.memory_reserved')
def test_memory_monitor_initialization(mock_reserved, mock_allocated, mock_is_available):
    """Test that GPUMemoryMonitor can be initialized."""
    from src.utils.memory_monitor import GPUMemoryMonitor
    
    # Configure the mocks
    mock_allocated.return_value = 1024**3  # 1GB
    mock_reserved.return_value = 2 * 1024**3  # 2GB
    
    # Initialize the monitor
    monitor = GPUMemoryMonitor()
    
    # Check that the monitor was initialized correctly
    assert monitor.thread is None
    assert monitor.keep_running is False
    assert monitor.interval == 60  # Default interval


@mock.patch('torch.cuda.is_available', return_value=True)
@mock.patch('torch.cuda.memory_allocated')
@mock.patch('torch.cuda.memory_reserved')
@mock.patch('threading.Thread')
def test_memory_monitor_start_stop(mock_thread, mock_reserved, mock_allocated, mock_is_available):
    """Test that GPUMemoryMonitor.start() and stop() work correctly."""
    from src.utils.memory_monitor import GPUMemoryMonitor
    
    # Configure the mocks
    mock_allocated.return_value = 1024**3  # 1GB
    mock_reserved.return_value = 2 * 1024**3  # 2GB
    mock_thread_instance = mock.MagicMock()
    mock_thread.return_value = mock_thread_instance
    
    # Initialize the monitor
    monitor = GPUMemoryMonitor(interval_seconds=10)
    
    # Start monitoring
    monitor.start()
    
    # Check that the thread was started
    assert monitor.keep_running is True
    mock_thread.assert_called_once()
    mock_thread_instance.start.assert_called_once()
    
    # Stop monitoring
    monitor.stop()
    
    # Check that the thread was stopped
    assert monitor.keep_running is False
    mock_thread_instance.join.assert_called_once()


@mock.patch('torch.cuda.is_available', return_value=False)
def test_memory_monitor_no_cuda(mock_is_available):
    """Test that GPUMemoryMonitor handles the case when CUDA is not available."""
    from src.utils.memory_monitor import GPUMemoryMonitor
    
    # Initialize the monitor
    monitor = GPUMemoryMonitor()
    
    # Start monitoring (should not start a thread)
    monitor.start()
    
    # Check that no thread was started
    assert monitor.thread is None


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 