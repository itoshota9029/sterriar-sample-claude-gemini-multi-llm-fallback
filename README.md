# Claude + Gemini Multi-LLM Fallback

[![CI](https://github.com/itoshota9029/sterriar-sample-claude-gemini-multi-llm-fallback/actions/workflows/ci.yml/badge.svg)](https://github.com/itoshota9029/sterriar-sample-claude-gemini-multi-llm-fallback/actions/workflows/ci.yml)

Claude を一次、Gemini を二次にしたフォールバックパターン。レート制限・障害時の切替実装。

## Features

- **Primary provider**: Claude (Anthropic API)
- **Fallback provider**: Gemini (Google Generative AI)
- **Automatic retry**: Rate-limit (429) および一般的なエラー時に自動的にフォールバック
- **Simple interface**: 統一された `ChatClient` インターフェースで透過的に切り替え

## Installation

```bash
git clone https://github.com/itoshota9029/sterriar-sample-claude-gemini-multi-llm-fallback.git
cd sterriar-sample-claude-gemini-multi-llm-fallback
pip install -e .
```

## Usage

```python
from src.fallback_client import FallbackChatClient

# API キーを環境変数 ANTHROPIC_API_KEY, GOOGLE_API_KEY から取得
client = FallbackChatClient(
    anthropic_api_key="your-claude-key",
    google_api_key="your-gemini-key"
)

response = client.chat("Hello, what is the capital of France?")
print(response)
```

## Development

```bash
pip install -e ".[dev]"

# Lint & format
ruff check .
ruff format .

# Test
pytest
```

## Architecture

- **`claude_client.py`**: Claude (Anthropic) API ラッパー
- **`gemini_client.py`**: Gemini (Google) API ラッパー
- **`fallback_client.py`**: フォールバックロジックを実装するメインクライアント

## License

MIT
