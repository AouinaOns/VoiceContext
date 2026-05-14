import { z } from "zod";

export const getTurnContextSchema = z.object({
  audio_base64: z.string().optional(),
  audio_url: z.string().url().optional(),
  format: z.enum(["wav", "mp3", "m4a", "ogg", "webm", "mp4"]).optional(),
  language: z.string().optional(),
  include_intent: z.boolean().default(false),
  include_diarization: z.boolean().default(false),
});

export const turnContextTool = {
  name: "get_turn_context",
  description: "Main tool for voice agents. Returns transcription + emotion + paralinguistics in one call.",
  inputSchema: getTurnContextSchema,
};
