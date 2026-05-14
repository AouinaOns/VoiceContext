FROM node:20-slim AS builder
WORKDIR /app
COPY package*.json turbo.json ./
COPY packages/core/package.json ./packages/core/
COPY packages/mcp-server/package.json ./packages/mcp-server/
RUN npm ci
COPY packages/core/ ./packages/core/
COPY packages/mcp-server/ ./packages/mcp-server/
RUN npm run build

FROM node:20-slim AS runner
WORKDIR /app
ENV NODE_ENV=production
COPY --from=builder /app/packages/core/dist ./packages/core/dist
COPY --from=builder /app/packages/core/package.json ./packages/core/
COPY --from=builder /app/packages/mcp-server/dist ./packages/mcp-server/dist
COPY --from=builder /app/packages/mcp-server/package.json ./packages/mcp-server/
COPY --from=builder /app/node_modules ./node_modules
EXPOSE 3000
CMD ["node", "packages/mcp-server/dist/transports/http.js"]
