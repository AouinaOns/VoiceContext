import type { VoiceContextConfig } from "@voicecontext/core";

export function loadConfig(): VoiceContextConfig {
  return {
    stt: {
      provider: (process.env.STT_PROVIDER as any) || "whisper-local",
      model: process.env.STT_MODEL || "whisper-large-v3",
      apiKey: process.env.STT_API_KEY,
      language: process.env.STT_LANGUAGE,
      diarization: process.env.DIARIZATION === "true",
    },
    emotion: {
      provider: (process.env.EMOTION_PROVIDER as any) || "speechbrain-local",
      apiKey: process.env.EMOTION_API_KEY,
      enabled: process.env.EMOTION_ENABLED !== "false",
    },
    intent: {
      enabled: process.env.INTENT_ENABLED === "true",
      domain: process.env.INTENT_DOMAIN || "generic",
      llmApiKey: process.env.OPENAI_API_KEY,
    },
    memory: {
      windowSize: parseInt(process.env.MEMORY_WINDOW || "10"),
      backend: (process.env.MEMORY_BACKEND as any) || "memory",
      redisUrl: process.env.REDIS_URL,
    },
  };
}
