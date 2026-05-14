# VoiceContext — The Contextual Speech API for Voice Agents

> One MCP tool call → transcription + emotion + paralinguistics + intent.

## What it does

VoiceContext processes a speech turn and returns a rich context object that voice agents can use to reason about *what was said*, *how it was said*, and *what the speaker intends*.

```json
{
  "turn_id": "3f7a...",
  "transcription": { "text": "I'm really not happy with this service", "language": "en", "confidence": 0.94 },
  "emotion":        { "dominant": "anger", "valence": -0.71, "arousal": 0.68 },
  "paralinguistics":{ "speech_rate_wpm": 108, "hesitations": 1, "pauses_ms": 420 },
  "intent":         { "name": "complaint", "entities": { "topic": "service" } },
  "processing_ms":  340
}
```

## Packages

| Package | Description |
|---|---|
| `@voicecontext/core` | Engine, types, STT/emotion/NLU adapters |
| `@voicecontext/mcp-server` | MCP server (stdio + HTTP/SSE transports) |
| `@voicecontext/sdk` | Typed HTTP client for direct API access |

## MCP Tools

| Tool | Description |
|---|---|
| `get_turn_context` | **Main tool.** Full context for one speech turn |
| `transcribe_audio` | STT only, with word timestamps |
| `analyze_emotion` | Emotion detection from audio or text |
| `detect_intent` | Intent + entity extraction |
| `get_conversation_summary` | Multi-turn conversation summary |

## Quick start

```bash
# Install dependencies
npm install

# Copy and fill in your providers
cp .env.example .env

# Run MCP server (stdio — for Claude Code / local agents)
npm run mcp:stdio

# Run MCP server (HTTP/SSE — for remote deployment)
npm run mcp:http
```

## Claude Code integration

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "voicecontext": {
      "command": "npx",
      "args": ["-y", "@voicecontext/mcp-server"],
      "env": {
        "STT_PROVIDER": "whisper-local",
        "EMOTION_ENABLED": "true"
      }
    }
  }
}
```

## Providers

**STT:** Whisper (local, free) · Deepgram · AssemblyAI · OpenAI Whisper API

**Emotion:** SpeechBrain/wav2vec2 (local) · Hume AI · Deepgram

**Intent:** LLM-based (OpenAI/Anthropic) — optional

## Architecture

```
Audio Input (WebSocket / File / Telephony / IoT)
        ↓
  @voicecontext/core
  ├── STT (transcription + diarization)
  ├── Emotion (audio + text fusion)
  ├── Paralinguistics (rate, pauses, hesitations)
  └── Intent & NLU (optional)
        ↓
  @voicecontext/mcp-server
  ├── stdio transport  → Claude Code, local agents
  └── HTTP/SSE transport → Remote deployment, SaaS
        ↓
  Any MCP-compatible consumer
  (Claude, custom voice agents, dashboards, support bots…)
```

## License

MIT
