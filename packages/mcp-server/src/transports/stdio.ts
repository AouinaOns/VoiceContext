#!/usr/bin/env node
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { VoiceContextEngine } from "@voicecontext/core";
import { createMcpServer } from "../index.js";
import { loadConfig } from "../middleware/config.js";

const config = loadConfig();
const engine = new VoiceContextEngine(config);
const server = createMcpServer(engine);
const transport = new StdioServerTransport();

await server.connect(transport);
console.error("VoiceContext MCP server running on stdio");
