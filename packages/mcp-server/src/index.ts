import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { VoiceContextEngine } from "@voicecontext/core";
import { transcribeTool, emotionTool, turnContextTool, intentTool } from "./tools/index.js";

export function createMcpServer(engine: VoiceContextEngine) {
  const server = new McpServer({
    name: "voicecontext",
    version: "0.1.0",
  });

  server.tool(transcribeTool.name, transcribeTool.description, transcribeTool.inputSchema.shape, async (args) => {
    // TODO: implement
    return { content: [{ type: "text", text: JSON.stringify({}) }] };
  });

  server.tool(emotionTool.name, emotionTool.description, emotionTool.inputSchema.shape, async (args) => {
    return { content: [{ type: "text", text: JSON.stringify({}) }] };
  });

  server.tool(turnContextTool.name, turnContextTool.description, turnContextTool.inputSchema.shape, async (args) => {
    return { content: [{ type: "text", text: JSON.stringify({}) }] };
  });

  server.tool(intentTool.name, intentTool.description, intentTool.inputSchema.shape, async (args) => {
    return { content: [{ type: "text", text: JSON.stringify({}) }] };
  });

  return server;
}
