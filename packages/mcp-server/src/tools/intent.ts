import { z } from "zod";

export const detectIntentSchema = z.object({
  text: z.string(),
  domain: z.enum(["support", "medical", "ecommerce", "hospitality", "generic"]).default("generic"),
  context_turns: z.array(z.string()).optional(),
});

export const intentTool = {
  name: "detect_intent",
  description: "Extract intent and named entities from transcribed text.",
  inputSchema: detectIntentSchema,
};
