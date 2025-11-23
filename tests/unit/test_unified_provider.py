"""Unit tests for UnifiedLLMProvider."""

import pytest
from unittest.mock import Mock, patch
from llm.providers import UnifiedLLMProvider


class TestUnifiedLLMProviderInit:
    """Test UnifiedLLMProvider initialization."""

    def test_openai_provider_init(self):
        """Test OpenAI provider initialization."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            provider = UnifiedLLMProvider(provider="openai", model="gpt-4o-mini")
            assert provider.provider_type == "openai"
            assert provider.model == "gpt-4o-mini"
            assert provider.api_key == "test-key"

    def test_grok_provider_init(self):
        """Test Grok provider initialization."""
        with patch.dict('os.environ', {'XAI_API_KEY': 'test-key'}):
            provider = UnifiedLLMProvider(provider="grok", model="grok-4-1-fast-reasoning")
            assert provider.provider_type == "grok"
            assert provider.model == "grok-4-1-fast-reasoning"

    def test_anthropic_provider_init(self):
        """Test Anthropic provider initialization."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            provider = UnifiedLLMProvider(provider="anthropic", model="claude-sonnet-4-5-20250929")
            assert provider.provider_type == "anthropic"
            assert provider.model == "claude-sonnet-4-5-20250929"

    def test_gemini_provider_init(self):
        """Test Gemini provider initialization."""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            provider = UnifiedLLMProvider(provider="gemini", model="gemini-2.5-flash")
            assert provider.provider_type == "gemini"
            assert provider.model == "gemini-2.5-flash"

    def test_ollama_provider_init(self):
        """Test Ollama provider initialization."""
        provider = UnifiedLLMProvider(provider="ollama", model="llama2")
        assert provider.provider_type == "ollama"
        assert provider.model == "llama2"
        assert provider.base_url == "http://localhost:11434"

    def test_invalid_provider_raises_error(self):
        """Test invalid provider raises ValueError."""
        with pytest.raises(ValueError, match="Unknown provider"):
            UnifiedLLMProvider(provider="invalid", model="test")


class TestUnifiedLLMProviderInitialization:
    """Test provider-specific initialization."""

    @patch('openai.OpenAI')
    def test_openai_client_initialized(self, mock_openai):
        """Test OpenAI client is initialized."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            provider = UnifiedLLMProvider(provider="openai", model="gpt-4o-mini")
            mock_openai.assert_called_once_with(
                api_key='test-key',
                base_url='https://api.openai.com/v1'
            )

    @patch('openai.OpenAI')
    def test_grok_client_initialized(self, mock_openai):
        """Test Grok client is initialized with xAI endpoint."""
        with patch.dict('os.environ', {'XAI_API_KEY': 'test-key'}):
            provider = UnifiedLLMProvider(provider="grok", model="grok-4-1-fast-reasoning")
            mock_openai.assert_called_once_with(
                api_key='test-key',
                base_url='https://api.x.ai/v1'
            )

    @patch('anthropic.Anthropic')
    def test_anthropic_client_initialized(self, mock_anthropic):
        """Test Anthropic client is initialized."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            provider = UnifiedLLMProvider(provider="anthropic", model="claude-sonnet-4-5-20250929")
            mock_anthropic.assert_called_once_with(api_key='test-key')

    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_gemini_client_initialized(self, mock_model, mock_configure):
        """Test Gemini SDK is initialized."""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            provider = UnifiedLLMProvider(provider="gemini", model="gemini-2.5-flash")
            mock_configure.assert_called_once_with(api_key='test-key')
            mock_model.assert_called_once_with("gemini-2.5-flash")


class TestUnifiedLLMProviderAvailability:
    """Test provider availability checking."""

    def test_openai_available_with_key(self):
        """Test OpenAI provider is available with API key."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            with patch('openai.OpenAI'):
                provider = UnifiedLLMProvider(provider="openai", model="gpt-4o-mini")
                assert provider.is_available() is True

    def test_openai_unavailable_without_key(self):
        """Test OpenAI provider is unavailable without API key."""
        with patch.dict('os.environ', {}, clear=True):
            with patch('openai.OpenAI'):
                provider = UnifiedLLMProvider(provider="openai", model="gpt-4o-mini")
                assert provider.is_available() is False

    def test_grok_available_with_key(self):
        """Test Grok provider is available with API key."""
        with patch.dict('os.environ', {'XAI_API_KEY': 'test-key'}):
            with patch('openai.OpenAI'):
                provider = UnifiedLLMProvider(provider="grok", model="grok-4-1-fast-reasoning")
                assert provider.is_available() is True

    def test_grok_unavailable_without_key(self):
        """Test Grok provider is unavailable without API key."""
        with patch.dict('os.environ', {}, clear=True):
            with patch('openai.OpenAI'):
                provider = UnifiedLLMProvider(provider="grok", model="grok-4-1-fast-reasoning")
                assert provider.is_available() is False

    def test_anthropic_available_with_key(self):
        """Test Anthropic provider is available with API key."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            with patch('anthropic.Anthropic'):
                provider = UnifiedLLMProvider(provider="anthropic", model="claude-sonnet-4-5-20250929")
                assert provider.is_available() is True

    def test_anthropic_unavailable_without_key(self):
        """Test Anthropic provider is unavailable without API key."""
        with patch.dict('os.environ', {}, clear=True):
            with patch('anthropic.Anthropic'):
                provider = UnifiedLLMProvider(provider="anthropic", model="claude-sonnet-4-5-20250929")
                assert provider.is_available() is False

    def test_gemini_available_with_key(self):
        """Test Gemini provider is available with API key."""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            with patch('google.generativeai.configure'):
                with patch('google.generativeai.GenerativeModel'):
                    provider = UnifiedLLMProvider(provider="gemini", model="gemini-2.5-flash")
                    assert provider.is_available() is True

    def test_gemini_unavailable_without_key(self):
        """Test Gemini provider is unavailable without API key."""
        with patch.dict('os.environ', {}, clear=True):
            with patch('google.generativeai.configure'):
                with patch('google.generativeai.GenerativeModel'):
                    provider = UnifiedLLMProvider(provider="gemini", model="gemini-2.5-flash")
                    assert provider.is_available() is False

    def test_ollama_available_when_server_running(self):
        """Test Ollama is available when server responds."""
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            provider = UnifiedLLMProvider(provider="ollama", model="llama2")
            assert provider.is_available() is True

    def test_ollama_unavailable_when_server_not_running(self):
        """Test Ollama is unavailable when server doesn't respond."""
        with patch('requests.get', side_effect=Exception("Connection refused")):
            provider = UnifiedLLMProvider(provider="ollama", model="llama2")
            assert provider.is_available() is False


class TestOpenAICompatibleGeneration:
    """Test response generation for OpenAI-compatible providers."""

    @patch('openai.OpenAI')
    def test_openai_generate_response(self, mock_openai_class):
        """Test OpenAI response generation."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            provider = UnifiedLLMProvider(provider="openai", model="gpt-4o-mini")
            response = provider.generate_response(
                prompt="Test prompt",
                system_prompt="Test system"
            )

            assert response == "Test response"
            mock_client.chat.completions.create.assert_called_once()
            call_args = mock_client.chat.completions.create.call_args[1]
            assert call_args["model"] == "gpt-4o-mini"
            assert len(call_args["messages"]) == 2
            assert call_args["messages"][0]["role"] == "system"
            assert call_args["messages"][1]["role"] == "user"

    @patch('openai.OpenAI')
    def test_grok_generate_response(self, mock_openai_class):
        """Test Grok response generation (OpenAI-compatible)."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Grok response"))]
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict('os.environ', {'XAI_API_KEY': 'test-key'}):
            provider = UnifiedLLMProvider(provider="grok", model="grok-4-1-fast-reasoning")
            response = provider.generate_response(prompt="Test")

            assert response == "Grok response"

    @patch('openai.OpenAI')
    def test_openai_json_mode(self, mock_openai_class):
        """Test OpenAI JSON mode."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='{"key": "value"}'))]
        mock_client.chat.completions.create.return_value = mock_response

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            provider = UnifiedLLMProvider(provider="openai", model="gpt-4o-mini")
            response = provider.generate_response(prompt="Test", force_json=True)

            call_args = mock_client.chat.completions.create.call_args[1]
            assert call_args["response_format"] == {"type": "json_object"}


class TestAnthropicGeneration:
    """Test Anthropic response generation."""

    @patch('anthropic.Anthropic')
    def test_anthropic_generate_response(self, mock_anthropic_class):
        """Test Anthropic response generation."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_response = Mock()
        mock_response.content = [Mock(text="Claude response")]
        mock_client.messages.create.return_value = mock_response

        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            provider = UnifiedLLMProvider(provider="anthropic", model="claude-sonnet-4-5-20250929")
            response = provider.generate_response(
                prompt="Test prompt",
                system_prompt="Test system"
            )

            assert response == "Claude response"
            mock_client.messages.create.assert_called_once()
            call_args = mock_client.messages.create.call_args[1]
            assert call_args["model"] == "claude-sonnet-4-5-20250929"
            assert call_args["system"] == "Test system"
            assert call_args["max_tokens"] == 4096

    @patch('anthropic.Anthropic')
    def test_anthropic_without_system_prompt(self, mock_anthropic_class):
        """Test Anthropic without system prompt."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client
        mock_response = Mock()
        mock_response.content = [Mock(text="Response")]
        mock_client.messages.create.return_value = mock_response

        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            provider = UnifiedLLMProvider(provider="anthropic", model="claude-sonnet-4-5-20250929")
            response = provider.generate_response(prompt="Test")

            call_args = mock_client.messages.create.call_args[1]
            assert "system" not in call_args or call_args.get("system") is None


class TestGeminiGeneration:
    """Test Gemini response generation."""

    @patch('google.generativeai.GenerativeModel')
    @patch('google.generativeai.configure')
    def test_gemini_generate_response(self, mock_configure, mock_model_class):
        """Test Gemini response generation."""
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        mock_response = Mock()
        mock_response.text = "Gemini response"
        mock_model.generate_content.return_value = mock_response

        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            provider = UnifiedLLMProvider(provider="gemini", model="gemini-2.5-flash")
            response = provider.generate_response(
                prompt="Test prompt",
                system_prompt="Test system"
            )

            assert response == "Gemini response"
            mock_model.generate_content.assert_called_once()

    @patch('google.generativeai.GenerativeModel')
    @patch('google.generativeai.configure')
    def test_gemini_json_mode(self, mock_configure, mock_model_class):
        """Test Gemini JSON mode."""
        mock_model = Mock()
        mock_model_class.return_value = mock_model
        mock_response = Mock()
        mock_response.text = '{"key": "value"}'
        mock_model.generate_content.return_value = mock_response

        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            provider = UnifiedLLMProvider(provider="gemini", model="gemini-2.5-flash")
            response = provider.generate_response(prompt="Test", force_json=True)

            call_args = mock_model.generate_content.call_args[1]
            assert call_args["generation_config"]["response_mime_type"] == "application/json"


class TestOllamaGeneration:
    """Test Ollama response generation."""

    def test_ollama_generate_response(self):
        """Test Ollama response generation."""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "message": {"content": "Ollama response"}
            }
            mock_post.return_value = mock_response

            with patch('requests.get') as mock_get:
                mock_get.return_value.status_code = 200  # is_available check

                provider = UnifiedLLMProvider(provider="ollama", model="llama2")
                response = provider.generate_response(
                    prompt="Test prompt",
                    system_prompt="Test system"
                )

                assert response == "Ollama response"
                mock_post.assert_called_once()
                call_args = mock_post.call_args[1]["json"]
                assert call_args["model"] == "llama2"
                assert len(call_args["messages"]) == 2

    def test_ollama_model_not_found_error(self):
        """Test Ollama model not found error."""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_response.raise_for_status.side_effect = Exception("404")
            mock_post.return_value = mock_response

            with patch('requests.get') as mock_get:
                mock_get.return_value.status_code = 200

                provider = UnifiedLLMProvider(provider="ollama", model="nonexistent")
                with pytest.raises(Exception, match="not found in Ollama"):
                    provider.generate_response(prompt="Test")
