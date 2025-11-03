import os
import pytest
from unittest.mock import Mock, patch, call
from datapipeline.scripts.train_test_val import GoodreadsNormalizedSplitter


@pytest.fixture
def mock_env():
    """Set up environment variables for testing."""
    with patch.dict(os.environ, {"AIRFLOW_HOME": "/test/airflow"}):
        yield


class TestGoodreadsNormalizedSplitter:
    """Test cases for GoodreadsNormalizedSplitter class."""

    @patch("datapipeline.scripts.train_test_val.get_logger")
    @patch("datapipeline.scripts.train_test_val.bigquery.Client")
    def test_init(self, mock_bq_client_class, mock_get_logger, mock_env):
        """Test initialization."""
        # Setup
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_bq_client_class.return_value = mock_client

        # Create instance
        splitter = GoodreadsNormalizedSplitter()

        # Assert
        # assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == "/test/airflow/gcp_credentials.json"
        mock_get_logger.assert_called_once_with("normalized_split")
        assert splitter.project_id == "test-project"
        assert splitter.dataset_id == "books"

    @patch("datapipeline.scripts.train_test_val.get_logger")
    @patch("datapipeline.scripts.train_test_val.bigquery.Client")
    def test_run_split_default_ratios(self, mock_bq_client_class, mock_get_logger, mock_env):
        """Test split with default 70/15/15 ratios."""
        # Setup
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_bq_client_class.return_value = mock_client
        mock_query_job = Mock()
        mock_client.query.return_value = mock_query_job

        # Execute
        splitter = GoodreadsNormalizedSplitter()
        splitter.run_split_in_bq()

        # Assert - verify 3 queries were made
        assert mock_client.query.call_count == 3

        # Check query boundaries
        queries = [call[0][0] for call in mock_client.query.call_args_list]

        assert "< 70" in queries[0]  # Train: 0-69
        assert "BETWEEN 70 AND 84" in queries[1]  # Val: 70-84
        assert ">= 85" in queries[2]  # Test: 85-99

        # Verify all queries were executed
        assert mock_query_job.result.call_count == 3

    @patch("datapipeline.scripts.train_test_val.get_logger")
    @patch("datapipeline.scripts.train_test_val.bigquery.Client")
    def test_run_method(self, mock_bq_client_class, mock_get_logger, mock_env):
        """Test the run method calls run_split_in_bq."""
        # Setup
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_bq_client_class.return_value = mock_client

        # Execute
        splitter = GoodreadsNormalizedSplitter()
        with patch.object(splitter, 'run_split_in_bq') as mock_split:
            splitter.run()
            mock_split.assert_called_once_with()

    @patch("datapipeline.scripts.train_test_val.get_logger")
    @patch("datapipeline.scripts.train_test_val.bigquery.Client")
    def test_bigquery_error_handling(self, mock_bq_client_class, mock_get_logger, mock_env):
        """Test that BigQuery errors are propagated."""
        # Setup
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_client.query.side_effect = Exception("BigQuery error")
        mock_bq_client_class.return_value = mock_client

        # Execute and assert
        splitter = GoodreadsNormalizedSplitter()
        with pytest.raises(Exception, match="BigQuery error"):
            splitter.run_split_in_bq()


# Parametrized test for different split ratios
@pytest.mark.parametrize("train,val,test,expected", [
    (0.7, 0.15, 0.15, (70, 84, 85)),
    (0.6, 0.2, 0.2, (60, 79, 80)),
    (0.8, 0.1, 0.1, (80, 89, 90)),
])
@patch("datapipeline.scripts.train_test_val.get_logger")
@patch("datapipeline.scripts.train_test_val.bigquery.Client")
def test_custom_split_ratios(mock_bq_client_class, mock_get_logger, train, val, test, expected):
    """Test various split ratio combinations."""
    # Setup
    with patch.dict(os.environ, {"AIRFLOW_HOME": "/test/airflow"}):
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_client = Mock()
        mock_client.project = "test-project"
        mock_bq_client_class.return_value = mock_client
        mock_query_job = Mock()
        mock_client.query.return_value = mock_query_job

        # Execute
        splitter = GoodreadsNormalizedSplitter()
        splitter.run_split_in_bq(train_ratio=train, val_ratio=val, test_ratio=test)

        # Extract queries
        queries = [call[0][0] for call in mock_client.query.call_args_list]

        # Assert boundaries
        train_limit, val_end, test_start = expected
        assert f"< {train_limit}" in queries[0]
        assert f"AND {val_end}" in queries[1]
        assert f">= {test_start}" in queries[2]


# Test for the main function
@patch("datapipeline.scripts.train_test_val.GoodreadsNormalizedSplitter")
def test_main(mock_splitter_class):
    """Test main function creates and runs splitter."""
    mock_instance = Mock()
    mock_splitter_class.return_value = mock_instance

    from datapipeline.scripts.train_test_val import main
    main()

    mock_splitter_class.assert_called_once()
    mock_instance.run.assert_called_once()

if __name__ == "__main__":
    pytest.main([__file__, '-q'])