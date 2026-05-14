import { z } from "zod";

export const analyzeEmotionSchema = z.object({
  audio_base64: z.string().optional(),
  audio_url: z.string().url().optional(),
  text: z.string().optional(),
  per_segment: z.boolean().default(false),
});

export const emotionTool = {
  name: "analyze_emotion",
  description: "Detect emotions from audio or text. Returns valence, arousal, and per-emotion scores.",
  inputSchema: analyzeEmotionSchema,
};
