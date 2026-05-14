#!/usr/bin/env node
import express from "express";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import { VoiceContextEngine } from "@voicecontext/core";
import { createMcpServer } from "../index.js";
import { loadConfig } from "../middleware/config.js";

const app = express();
const config = loadConfig();
const engine = new VoiceContextEngine(config);
const server = createMcpServer(engine);

const PORT = process.env.PORT || 3000;
const transports: Record<string, SSEServerTransport> = {};

app.get("/sse", async (req, res) => {
  const transport = new SSEServerTransport("/messages", res);
  transports[transport.sessionId] = transport;
  await server.connect(transport);
});

app.post("/messages", async (req, res) => {
  const sessionId = req.query.sessionId as string;
  const transport = transports[sessionId];
  if (!transport) { res.status(404).send("Session not found"); return; }
  await transport.handlePostMessage(req, res);
});

app.listen(PORT, () => {
  console.log(`VoiceContext MCP server running on http://localhost:${PORT}`);
});
