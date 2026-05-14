import axios, { type AxiosInstance } from "axios";
import type { AudioInput, TurnContext, Transcription, EmotionResult, IntentResult } from "@voicecontext/core";

export interface VoiceContextClientOptions {
  baseUrl: string;
  apiKey?: string;
  timeout?: number;
}

export class VoiceContextClient {
  private http: AxiosInstance;

  constructor(options: VoiceContextClientOptions) {
    this.http = axios.create({
      baseURL: options.baseUrl,
      timeout: options.timeout || 30000,
      headers: options.apiKey ? { Authorization: `Bearer ${options.apiKey}` } : {},
    });
  }

  async getTurnContext(audio: AudioInput, opts?: { includeIntent?: boolean }): Promise<TurnContext> {
    const { data } = await this.http.post<TurnContext>("/v1/turn-context", { audio, ...opts });
    return data;
  }

  async transcribe(audio: AudioInput): Promise<Transcription> {
    const { data } = await this.http.post<Transcription>("/v1/transcribe", { audio });
    return data;
  }

  async analyzeEmotion(audio: AudioInput, text?: string): Promise<EmotionResult> {
    const { data } = await this.http.post<EmotionResult>("/v1/emotion", { audio, text });
    return data;
  }

  async detectIntent(text: string, domain = "generic"): Promise<IntentResult> {
    const { data } = await this.http.post<IntentResult>("/v1/intent", { text, domain });
    return data;
  }
}
