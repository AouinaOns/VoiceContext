import { randomUUID } from "crypto";
import type { AudioInput, TurnContext, VoiceContextConfig } from "./types/index.js";
import { transcribe } from "./stt/index.js";
import { analyzeEmotion } from "./emotion/index.js";
import { extractParalinguistics } from "./paralinguistics/index.js";
import { detectIntent } from "./intent/index.js";

export class VoiceContextEngine {
  constructor(private config: VoiceContextConfig) {}

  async processAudio(input: AudioInput): Promise<TurnContext> {
    const start = Date.now();
    const turn_id = randomUUID();

    // STT (always required)
    const transcription = await transcribe(input, this.config.stt);

    // Parallel: emotion + paralinguistics (don't block on each other)
    const [emotion, paralinguistics] = await Promise.all([
      this.config.emotion.enabled
        ? analyzeEmotion(input, transcription, this.config.emotion)
        : Promise.resolve(defaultEmotion()),
      extractParalinguistics(transcription),
    ]);

    // Intent (optional, LLM call)
    const intent = this.config.intent?.enabled
      ? await detectIntent(transcription.text, this.config.intent)
      : undefined;

    return {
      turn_id,
      timestamp: new Date().toISOString(),
      transcription,
      emotion,
      paralinguistics,
      intent,
      speakers: [...new Set(transcription.segments.map((s) => s.speaker).filter(Boolean) as string[])],
      processing_ms: Date.now() - start,
    };
  }
}

function defaultEmotion() {
  return {
    dominant: "neutral" as const,
    valence: 0,
    arousal: 0,
    confidence: 0,
    scores: [],
  };
}
