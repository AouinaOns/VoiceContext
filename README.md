# VoiceContext

> A personal side project — I'm building this in my spare time to explore LLM agents and voice AI.

The idea: one MCP tool call that returns everything a voice agent needs to understand a speech turn — transcription, emotion, paralinguistics, and intent.

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

## Status

Work in progress. I feed it whenever I have time. Things may break, change, or disappear. If you're curious or want to follow along, feel free — issues and ideas welcome.

Current experiments:
- [x] Basic agent loop with tool use
- [x] NeMo streaming integration (see `notebooks/03_nemo_streaming_test.ipynb`)
- [ ] Emotion detection pipeline
- [ ] MCP server wiring

## What it does

VoiceContext processes a speech turn and returns a rich context object so a voice agent can reason about *what was said*, *how it was said*, and *what the speaker intends*.

## Architecture

```
Audio Input (WebSocket / File / Telephony)
        ↓
  core engine
  ├── STT (transcription + diarization)
  ├── Emotion (audio + text fusion)
  ├── Paralinguistics (rate, pauses, hesitations)
  └── Intent & NLU (optional)
        ↓
  MCP server
  ├── stdio transport  → Claude Code, local agents
  └── HTTP/SSE transport → remote deployment
        ↓
  Any MCP-compatible consumer
```

## MCP Tools (planned)

| Tool | Description |
|---|---|
| `get_turn_context` | Full context for one speech turn |
| `transcribe_audio` | STT only, with word timestamps |
| `analyze_emotion` | Emotion detection from audio or text |
| `detect_intent` | Intent + entity extraction |
| `get_conversation_summary` | Multi-turn conversation summary |

## Quick start

```bash
# Python demo agent
cd apps/demo
uv sync
uv run src/simple_agent.py
```

## Providers explored

**STT:** Whisper (local) · NeMo · Deepgram · AssemblyAI

**Emotion:** SpeechBrain/wav2vec2 · Hume AI

**Intent:** LLM-based (OpenAI / Anthropic)

## Claude Code integration

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

## License

Made with &lt;3 — do whatever you want with it.
