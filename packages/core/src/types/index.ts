export type AudioFormat = "wav" | "mp3" | "m4a" | "ogg" | "webm" | "mp4";
export type AudioSource = "file" | "stream" | "url" | "base64";

export interface AudioInput {
  source: AudioSource;
  data: Buffer | string;
  format?: AudioFormat;
  sampleRate?: number;
  language?: string;
}

export interface TranscriptWord {
  word: string;
  start: number;
  end: number;
  confidence: number;
  speaker?: string;
}

export interface TranscriptSegment {
  id: number;
  text: string;
  start: number;
  end: number;
  speaker?: string;
  words?: TranscriptWord[];
}

export interface Transcription {
  text: string;
  language: string;
  duration: number;
  confidence: number;
  segments: TranscriptSegment[];
  words?: TranscriptWord[];
}

export type EmotionLabel =
  | "joy" | "anger" | "sadness" | "fear"
  | "surprise" | "disgust" | "neutral" | "anxiety";

export interface EmotionScore {
  label: EmotionLabel;
  score: number;
}

export interface EmotionResult {
  dominant: EmotionLabel;
  valence: number;
  arousal: number;
  confidence: number;
  scores: EmotionScore[];
  per_segment?: Array<{
    segment_id: number;
    dominant: EmotionLabel;
    valence: number;
    arousal: number;
  }>;
}

export interface ParalinguisticFeatures {
  speech_rate_wpm: number;
  hesitations: number;
  pauses_ms: number;
  avg_pause_ms: number;
  intensity_db?: number;
  pitch_hz?: number;
  laughter: boolean;
}

export interface IntentEntity {
  name: string;
  value: string;
  confidence: number;
}

export interface IntentResult {
  name: string;
  confidence: number;
  entities: Record<string, string>;
  entities_detail?: IntentEntity[];
}

export interface TurnContext {
  turn_id: string;
  timestamp: string;
  transcription: Transcription;
  emotion: EmotionResult;
  paralinguistics: ParalinguisticFeatures;
  intent?: IntentResult;
  speakers?: string[];
  processing_ms: number;
}

export type STTProvider = "whisper-local" | "deepgram" | "assemblyai" | "openai";
export type EmotionProvider = "speechbrain-local" | "hume" | "deepgram";

export interface VoiceContextConfig {
  stt: {
    provider: STTProvider;
    model?: string;
    apiKey?: string;
    language?: string;
    diarization?: boolean;
  };
  emotion: {
    provider: EmotionProvider;
    apiKey?: string;
    enabled: boolean;
  };
  intent?: {
    enabled: boolean;
    domain?: string;
    llmApiKey?: string;
  };
  memory?: {
    windowSize?: number;
    backend?: "memory" | "redis";
    redisUrl?: string;
  };
}
