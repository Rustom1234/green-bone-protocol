import os
import pytest
from unittest.mock import patch, MagicMock
import anthropic
from main import get_copenhagen_tattoo_recommendations


class TestGetCopenhagenTattooRecommendations:
    """Test suite for get_copenhagen_tattoo_recommendations function."""

    def test_returns_string(self):
        """Test that function returns a string."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Sample recommendations")]
        
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as mock_client_class:
                mock_instance = MagicMock()
                mock_instance.messages.create.return_value = mock_response
                mock_client_class.return_value = mock_instance
                
                result = get_copenhagen_tattoo_recommendations()
                assert isinstance(result, str)

    def test_uses_correct_api_key(self):
        """Test that function uses ANTHROPIC_API_KEY from environment."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Recommendations")]
        
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "my-test-key"}):
            with patch("anthropic.Anthropic") as mock_client_class:
                mock_instance = MagicMock()
                mock_instance.messages.create.return_value = mock_response
                mock_client_class.return_value = mock_instance
                
                get_copenhagen_tattoo_recommendations()
                mock_client_class.assert_called_once_with(api_key="my-test-key")

    def test_api_key_missing_raises_key_error(self):
        """Test that missing API key raises KeyError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(KeyError):
                get_copenhagen_tattoo_recommendations()

    def test_calls_messages_create_with_correct_parameters(self):
        """Test that messages.create is called with correct model and max_tokens."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Recommendations")]
        
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as mock_client_class:
                mock_instance = MagicMock()
                mock_instance.messages.create.return_value = mock_response
                mock_client_class.return_value = mock_instance
                
                get_copenhagen_tattoo_recommendations()
                
                call_args = mock_instance.messages.create.call_args
                assert call_args.kwargs["model"] == "claude-3-5-haiku-20241022"
                assert call_args.kwargs["max_tokens"] == 1024

    def test_messages_create_receives_correct_prompt(self):
        """Test that the prompt contains expected keywords."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Recommendations")]
        
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as mock_client_class:
                mock_instance = MagicMock()
                mock_instance.messages.create.return_value = mock_response
                mock_client_class.return_value = mock_instance
                
                get_copenhagen_tattoo_recommendations()
                
                call_args = mock_instance.messages.create.call_args
                messages = call_args.kwargs["messages"]
                assert len(messages) == 1
                assert messages[0]["role"] == "user"
                content = messages[0]["content"]
                assert "Copenhagen" in content
                assert "tattoo" in content
                assert "5" in content

    def test_extracts_text_from_response_content(self):
        """Test that function correctly extracts text from response."""
        expected_text = "1. Shop A\n2. Shop B\n3. Shop C\n4. Shop D\n5. Shop E"
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=expected_text)]
        
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as mock_client_class:
                mock_instance = MagicMock()
                mock_instance.messages.create.return_value = mock_response
                mock_client_class.return_value = mock_instance
                
                result = get_copenhagen_tattoo_recommendations()
                assert result == expected_text

    def test_returns_empty_string_when_response_empty(self):
        """Test handling of empty response text."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="")]
        
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as mock_client_class:
                mock_instance = MagicMock()
                mock_instance.messages.create.return_value = mock_response
                mock_client_class.return_value = mock_instance
                
                result = get_copenhagen_tattoo_recommendations()
                assert result == ""

    def test_handles_multiline_response(self):
        """Test that function handles multiline responses correctly."""
        multiline_text = "Shop 1\n  Neighborhood: A\n  Style: B\n\nShop 2\n  Neighborhood: C\n  Style: D"
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=multiline_text)]
        
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as mock_client_class:
                mock_instance = MagicMock()
                mock_instance.messages.create.return_value = mock_response
                mock_client_class.return_value = mock_instance
                
                result = get_copenhagen_tattoo_recommendations()
                assert "\n" in result
                assert result == multiline_text

    def test_client_instantiation_called_once(self):
        """Test that Anthropic client is instantiated exactly once."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Recommendations")]
        
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as mock_client_class:
                mock_instance = MagicMock()
                mock_instance.messages.create.return_value = mock_response
                mock_client_class.return_value = mock_instance
                
                get_copenhagen_tattoo_recommendations()
                assert mock_client_class.call_count == 1

    def test_response_content_first_element_accessed(self):
        """Test that function accesses content[0] from response."""
        mock_content_obj = MagicMock()
        mock_content_obj.text = "Test response"
        mock_response = MagicMock()
        mock_response.content = [mock_content_obj]
        
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("anthropic.Anthropic") as mock_client_class:
                mock_instance = MagicMock()
                mock_instance.messages.create.return_value = mock_response
                mock_client_class.return_value = mock_instance
                
                result = get_copenhagen_tattoo_recommendations()
                assert result == "Test response"

    def test_with_long_response(self