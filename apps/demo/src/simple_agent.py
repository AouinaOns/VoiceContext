"""
simple_agent.py
---------------
The simplest possible voice agent on top of VoiceContext.

Architecture:
  Audio → VoiceContext (NeMo STT + emotion + paralinguistics + intent)
        → TurnContext (one object)
        → LLM (Claude) reasons on TurnContext + history
        → decides: reply text OR tool call
        → loops

Run:
  python apps/demo/simple_agent.py --mic
  python apps/demo/simple_agent.py --audio path/to/file.wav
  python apps/demo/simple_agent.py --text   # text-only mode, skip audio

Dependencies:
  pip install anthropic python-dotenv
  (VoiceContext core must be installed or in PYTHONPATH)
"""

import os
import json
import time
import argparse
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# ─── TurnContext stub (replace with real VoiceContext once NeMo is set up) ────

@dataclass
class Transcription:
    text: str
    language: str = "en"
    confidence: float = 0.95
    duration: float = 0.0

@dataclass
class EmotionResult:
    dominant: str = "neutral"
    valence: float = 0.0      # -1 (negative) to +1 (positive)
    arousal: float = 0.0      # 0 (calm) to 1 (excited)
    confidence: float = 0.8

@dataclass
class ParalinguisticFeatures:
    speech_rate_wpm: int = 130
    hesitations: int = 0
    pauses_ms: int = 0
    laughter: bool = False

@dataclass
class IntentResult:
    name: str = "unknown"
    confidence: float = 0.0
    entities: dict = field(default_factory=dict)

@dataclass
class TurnContext:
    turn_id: str
    timestamp: str
    transcription: Transcription
    emotion: EmotionResult
    paralinguistics: ParalinguisticFeatures
    intent: Optional[IntentResult] = None
    processing_ms: float = 0.0

    def to_prompt_str(self) -> str:
        """Serialise TurnContext into a compact string the LLM can parse."""
        parts = [
            f'Text: "{self.transcription.text}"',
            f"Language: {self.transcription.language}",
            f"Emotion: {self.emotion.dominant} (valence={self.emotion.valence:+.2f}, arousal={self.emotion.arousal:.2f})",
            f"Speech rate: {self.paralinguistics.speech_rate_wpm} wpm",
        ]
        if self.paralinguistics.hesitations > 0:
            parts.append(f"Hesitations: {self.paralinguistics.hesitations}")
        if self.paralinguistics.laughter:
            parts.append("Laughter detected")
        if self.intent and self.intent.confidence > 0.5:
            parts.append(f"Intent: {self.intent.name} (conf={self.intent.confidence:.2f})")
            if self.intent.entities:
                parts.append(f"Entities: {json.dumps(self.intent.entities)}")
        return "\n".join(parts)


# ─── Audio / STT (stub — swap with real NeMo engine) ─────────────────────────

def perceive_from_text(text: str) -> TurnContext:
    """Simulate a TurnContext from raw text (for testing without audio)."""
    import uuid
    from datetime import datetime, timezone

    # Simple heuristic emotion detection for demo
    negative_words = ["frustrated", "angry", "hate", "terrible", "awful", "bad", "wrong", "problem"]
    positive_words = ["great", "thanks", "love", "amazing", "perfect", "excellent", "good"]
    hesitation_words = ["uh", "um", "euh", "hmm", "er"]

    text_lower = text.lower()
    neg_count = sum(1 for w in negative_words if w in text_lower)
    pos_count = sum(1 for w in positive_words if w in text_lower)
    hes_count = sum(1 for w in hesitation_words if w in text_lower)

    valence = min(1, max(-1, (pos_count - neg_count) * 0.35))
    arousal = min(1, neg_count * 0.3 + pos_count * 0.1)
    dominant = "anger" if valence < -0.3 else "joy" if valence > 0.3 else "neutral"

    word_count = len(text.split())
    intent_name = "complaint" if neg_count > 0 else "inquiry" if "?" in text else "statement"

    return TurnContext(
        turn_id=str(uuid.uuid4())[:8],
        timestamp=datetime.now(timezone.utc).isoformat(),
        transcription=Transcription(text=text, confidence=1.0, duration=word_count / 2.5),
        emotion=EmotionResult(dominant=dominant, valence=valence, arousal=arousal),
        paralinguistics=ParalinguisticFeatures(
            speech_rate_wpm=int(word_count / max(0.01, word_count / 130)),
            hesitations=hes_count,
        ),
        intent=IntentResult(name=intent_name, confidence=0.75),
    )


def perceive_from_audio(audio_path: str) -> TurnContext:
    """
    Real NeMo-based perception. Replace stub with actual VoiceContext engine.
    This will call: NeMo FastConformer (STT) + SpeechBrain (emotion) + paralinguistics.
    """
    try:
        # This will work once VoiceContext core is implemented
        from voicecontext.core.engine import VoiceContextEngine
        from voicecontext.core.types import AudioInput
        engine = VoiceContextEngine.from_env()
        audio_input = AudioInput(source="file", data=audio_path)
        return engine.process_audio(audio_input)
    except ImportError:
        # Fallback: run NeMo STT and build TurnContext manually
        print("  [VoiceContext core not installed — running NeMo directly]")
        return _perceive_nemo_direct(audio_path)


def _perceive_nemo_direct(audio_path: str) -> TurnContext:
    """Direct NeMo call. Used until VoiceContext core is packaged."""
    import uuid
    import os
    from datetime import datetime, timezone
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    import torch
    import nemo.collections.asr as nemo_asr

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
        "nvidia/nemotron-speech-streaming-en-0.6b"
    )
    model = model.to(device)
    model.eval()

    t0 = time.perf_counter()
    with torch.no_grad():
        result = model.transcribe([audio_path])
    elapsed = (time.perf_counter() - t0) * 1000

    text = result[0].text if hasattr(result[0], "text") else str(result[0])

    return TurnContext(
        turn_id=str(uuid.uuid4())[:8],
        timestamp=datetime.now(timezone.utc).isoformat(),
        transcription=Transcription(text=text, confidence=0.92),
        emotion=EmotionResult(),       # TODO: add SpeechBrain
        paralinguistics=ParalinguisticFeatures(),  # TODO: extract from audio
        processing_ms=elapsed,
    )


# ─── Tools the agent can call ─────────────────────────────────────────────────

TOOLS = [
    {
        "name": "escalate_to_human",
        "description": "Escalate this conversation to a human agent when the user is highly distressed or has a complex problem the AI cannot resolve.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Why escalation is needed"},
                "priority": {"type": "string", "enum": ["low", "normal", "high", "urgent"]},
            },
            "required": ["reason", "priority"],
        },
    },
    {
        "name": "search_knowledge_base",
        "description": "Search the knowledge base for information relevant to the user's question.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "create_support_ticket",
        "description": "Create a support ticket for follow-up.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "priority": {"type": "string", "enum": ["low", "normal", "high"]},
            },
            "required": ["title", "description", "priority"],
        },
    },
]


def execute_tool(tool_name: str, tool_input: dict) -> dict:
    """Stub tool executors. Replace with real implementations."""
    print(f"\n  ⚙  Calling tool: {tool_name}")
    print(f"     Input: {json.dumps(tool_input, indent=6)}")

    if tool_name == "escalate_to_human":
        return {
            "success": True,
            "ticket_id": "ESC-2847",
            "eta_minutes": 3,
            "message": f"Escalated to human agent. ETA: 3 minutes. Reason: {tool_input['reason']}",
        }
    if tool_name == "search_knowledge_base":
        return {
            "results": [
                {"title": "How to reset your password", "url": "/kb/reset-password", "relevance": 0.91},
                {"title": "Account billing FAQ", "url": "/kb/billing-faq", "relevance": 0.72},
            ],
            "query": tool_input["query"],
        }
    if tool_name == "create_support_ticket":
        return {
            "success": True,
            "ticket_id": f"TICKET-{int(time.time()) % 10000}",
            "message": f"Ticket created: {tool_input['title']}",
        }
    return {"error": f"Unknown tool: {tool_name}"}


# ─── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a voice support agent for a software company.

You receive each user turn as a structured TurnContext object containing:
- Text: the transcribed speech
- Emotion: dominant emotion, valence (-1=very negative, +1=very positive), arousal (0=calm, 1=excited)
- Speech rate: words per minute (slow < 100, normal 100-160, fast > 160)
- Hesitations: count of "um", "uh", "euh" etc.
- Intent: classified intent (complaint, inquiry, statement, etc.)

Use ALL of this context to respond appropriately:
- If valence < -0.5 AND intent == "complaint": escalate or acknowledge strongly
- If hesitations > 2: the user seems uncertain — be extra clear and patient
- If speech rate > 180: the user is rushed — be concise
- If laughter is detected: mirror their positive energy
- Always acknowledge the emotional tone before addressing the content
- Keep responses conversational and brief (1-3 sentences). You are a voice agent.

Available tools: escalate_to_human, search_knowledge_base, create_support_ticket
"""


# ─── Agent loop ───────────────────────────────────────────────────────────────

class VoiceAgent:
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-6"):
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        except ImportError:
            raise RuntimeError("Install anthropic: pip install anthropic")

        self.model = model
        self.history: list[dict] = []
        self.turn_count = 0

    def process_turn(self, ctx: TurnContext) -> str:
        """
        One full agent loop iteration.
        Perceive (ctx) → Think (LLM) → Decide (tool|reply) → Act → return response text.
        """
        self.turn_count += 1

        # ── 1. Build user message from TurnContext ──────────────────────────
        user_content = f"[Turn {self.turn_count} — {ctx.turn_id}]\n{ctx.to_prompt_str()}"
        self.history.append({"role": "user", "content": user_content})

        # ── 2. LLM thinks ───────────────────────────────────────────────────
        import anthropic
        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=self.history,
        )

        # ── 3. Decide + Act ─────────────────────────────────────────────────
        final_text = ""

        while response.stop_reason == "tool_use":
            # The LLM decided to call a tool
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    })

            # Append LLM turn + tool results to history
            self.history.append({"role": "assistant", "content": response.content})
            self.history.append({"role": "user", "content": tool_results})

            # LLM continues reasoning with tool results
            response = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=self.history,
            )

        # Extract final text reply
        for block in response.content:
            if hasattr(block, "text"):
                final_text = block.text
                break

        # ── 4. Append assistant reply to history ────────────────────────────
        self.history.append({"role": "assistant", "content": final_text})
        return final_text

    def reset(self):
        """Start a new conversation."""
        self.history = []
        self.turn_count = 0
        print("  [New conversation started]")


# ─── Demo modes ───────────────────────────────────────────────────────────────

def run_text_mode(agent: VoiceAgent):
    """Interactive text-input demo. No audio needed."""
    print("\n" + "═" * 60)
    print("  VoiceContext Agent — Text demo mode")
    print("  (Type your message, or 'quit' to exit, 'reset' for new session)")
    print("═" * 60)

    while True:
        try:
            user_input = input("\n🎤  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nSession ended.")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            agent.reset()
            continue

        # Perceive: text → TurnContext
        t0 = time.perf_counter()
        ctx = perceive_from_text(user_input)
        perceive_ms = (time.perf_counter() - t0) * 1000

        print(f"\n  📊  Context: emotion={ctx.emotion.dominant} (valence={ctx.emotion.valence:+.2f}), "
              f"intent={ctx.intent.name if ctx.intent else 'n/a'}, "
              f"hesitations={ctx.paralinguistics.hesitations}")

        # Think + Decide + Act
        t0 = time.perf_counter()
        response = agent.process_turn(ctx)
        llm_ms = (time.perf_counter() - t0) * 1000

        print(f"\n🤖  Agent: {response}")
        print(f"\n  ⏱  perceive={perceive_ms:.0f}ms | llm={llm_ms:.0f}ms | "
              f"turns in history={len(agent.history)}")


def run_mic_mode(agent: VoiceAgent, chunk_ms: int = 160):
    """Live microphone mode using NeMo streaming."""
    import os
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    import numpy as np
    import torch
    import pyaudio
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

    SAMPLE_RATE = 16000
    chunk_samples = int(SAMPLE_RATE * chunk_ms / 1000)

    print(f"\nLoading NeMo model (chunk={chunk_ms}ms)...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
        "nvidia/nemotron-speech-streaming-en-0.6b"
    )
    model = model.to(device).eval()

    streaming_buffer = CacheAwareStreamingAudioBuffer(model=model, online_normalization=True)
    cache = streaming_buffer.get_buffers_state()

    # Warmup pass
    silence = np.zeros(chunk_samples, dtype=np.float32)
    with torch.no_grad():
        streaming_buffer.process_chunk(silence, *cache)
    cache = streaming_buffer.get_buffers_state()

    pa = pyaudio.PyAudio()
    mic = pa.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE,
                  input=True, frames_per_buffer=chunk_samples)

    print("\n" + "═" * 60)
    print("  VoiceContext Agent — Live mic mode")
    print(f"  Streaming chunk={chunk_ms}ms | Say something (Ctrl+C to quit)")
    print("═" * 60)

    # Simple VAD: accumulate chunks until 800ms of silence
    SILENCE_THRESHOLD = 0.01
    SILENCE_CHUNKS = int(800 / chunk_ms)

    accumulated_text = ""
    silence_count = 0
    speaking = False

    try:
        while True:
            raw = mic.read(chunk_samples, exception_on_overflow=False)
            chunk = np.frombuffer(raw, dtype=np.float32).copy()
            energy = float(np.abs(chunk).mean())

            with torch.no_grad():
                partial, *cache = streaming_buffer.process_chunk(chunk, *cache)

            if energy > SILENCE_THRESHOLD:
                speaking = True
                silence_count = 0
                if partial:
                    accumulated_text += partial
                    print(f"  🎙  {partial}", end="", flush=True)
            else:
                if speaking:
                    silence_count += 1
                    if silence_count >= SILENCE_CHUNKS and accumulated_text.strip():
                        # End of utterance detected
                        print()  # newline after streaming text
                        utterance = accumulated_text.strip()
                        accumulated_text = ""
                        speaking = False
                        silence_count = 0
                        cache = streaming_buffer.get_buffers_state()  # reset cache between turns

                        # Perceive → TurnContext
                        ctx = perceive_from_text(utterance)  # TODO: replace with full audio perception
                        print(f"\n  📊  emotion={ctx.emotion.dominant} (valence={ctx.emotion.valence:+.2f})")

                        # Think → Act
                        response = agent.process_turn(ctx)
                        print(f"\n🤖  Agent: {response}\n")

    except KeyboardInterrupt:
        print("\n\nSession ended.")
        mic.stop_stream()
        mic.close()
        pa.terminate()


def run_file_mode(agent: VoiceAgent, audio_path: str):
    """Process a single audio file."""
    print(f"\nProcessing: {audio_path}")
    ctx = perceive_from_audio(audio_path)
    print(f"\n  📊  Transcript: '{ctx.transcription.text}'")
    print(f"  📊  Emotion: {ctx.emotion.dominant} (valence={ctx.emotion.valence:+.2f})")
    response = agent.process_turn(ctx)
    print(f"\n🤖  Agent: {response}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="VoiceContext simple agent demo")
    p.add_argument("--text",     action="store_true", help="Text input mode (no audio needed)")
    p.add_argument("--mic",      action="store_true", help="Live microphone mode")
    p.add_argument("--audio",    type=str,            help="Process an audio file")
    p.add_argument("--model",    default="claude-sonnet-4-6", help="Claude model to use")
    p.add_argument("--chunk-ms", type=int, default=160, choices=[80, 160, 560, 1120])
    args = p.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set. Add it to your .env or export it.")
        return

    agent = VoiceAgent(model=args.model)

    if args.text or (not args.mic and not args.audio):
        run_text_mode(agent)
    elif args.mic:
        run_mic_mode(agent, chunk_ms=args.chunk_ms)
    elif args.audio:
        run_file_mode(agent, args.audio)


if __name__ == "__main__":
    main()
