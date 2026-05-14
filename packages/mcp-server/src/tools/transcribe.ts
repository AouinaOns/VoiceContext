import { z } from "zod";

export const transcribeSchema = z.object({
  audio_base64: z.string().optional(),
  audio_url: z.string().url().optional(),
  format: z.enum(["wav", "mp3", "m4a", "ogg", "webm", "mp4"]).optional(),
  language: z.string().optional(),
  diarization: z.boolean().default(false),
});

export const transcribeTool = {
  name: "transcribe_audio",
  description: "Transcribe audio to text with word-level timestamps and confidence scores.",
  inputSchema: transcribeSchema,
};
