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
