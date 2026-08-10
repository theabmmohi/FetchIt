import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js"
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js"
import express, { Request, Response } from "express"
import chalk from "chalk"
import { z } from "zod"

// ============================================================================
// Dev Logging Utilities (same pattern as mcp32's index.ts)
// ============================================================================

const isDev = process.env.NODE_ENV !== "production"

function timestamp(): string {
  return new Date().toLocaleTimeString("en-US", { hour12: false })
}

function formatLatency(ms: number): string {
  if (ms < 100) return chalk.green(`${ms}ms`)
  if (ms < 500) return chalk.yellow(`${ms}ms`)
  return chalk.red(`${ms}ms`)
}

function truncate(str: string, maxLen = 60): string {
  if (str.length <= maxLen) return str
  return str.slice(0, maxLen - 3) + "..."
}

function logRequest(method: string, params?: unknown): void {
  if (!isDev) return
  const paramsStr = params ? chalk.gray(` ${truncate(JSON.stringify(params))}`) : ""
  console.log(`${chalk.gray(`[${timestamp()}]`)} ${chalk.cyan("→")} ${method}${paramsStr}`)
}

function logResponse(method: string, result: unknown, latencyMs: number): void {
  if (!isDev) return
  const latency = formatLatency(latencyMs)
  if (method === "tools/call" && result) {
    const resultStr = typeof result === "string" ? result : JSON.stringify(result)
    console.log(`${chalk.gray(`[${timestamp()}]`)} ${chalk.green("←")} ${truncate(resultStr)} ${chalk.gray(`(${latency})`)}`)
  }
  else console.log(`${chalk.gray(`[${timestamp()}]`)} ${chalk.green("✓")} ${method} ${chalk.gray(`(${latency})`)}`)
}

function logError(method: string, error: unknown, latencyMs: number): void {
  const latency = formatLatency(latencyMs)
  let errorMsg: string
  if (error instanceof Error) errorMsg = error.message
  else if (typeof error === "object" && error !== null) {
    const rpcError = error as { message?: string; code?: number }
    errorMsg = rpcError.message || `Error ${rpcError.code || "unknown"}`
  }
  else errorMsg = String(error)
  console.log(`${chalk.gray(`[${timestamp()}]`)} ${chalk.red("✖")} ${method} ${chalk.red(truncate(errorMsg))} ${chalk.gray(`(${latency})`)}`)
}

// ============================================================================
// FetchIt tool logic
// ============================================================================

const MAX_BODY_CHARS = 50_000 // keep large responses from blowing up model context
const REQUEST_TIMEOUT_MS = 15_000

// Basic SSRF guard: blocks the obvious internal/metadata targets by literal
// hostname/IP. This does NOT resolve DNS, so a hostname that resolves to a
// private IP (DNS rebinding) would slip through — good enough for a personal
// testing tool, not a substitute for network-level egress controls if this
// is ever exposed more broadly.
const BLOCKED_HOSTS = new Set([
  "localhost",
  "127.0.0.1",
  "0.0.0.0",
  "::1",
  "169.254.169.254", // cloud metadata endpoint (AWS/GCP/Azure)
  "metadata.google.internal",
])

function isPrivateIp(host: string): boolean {
  const parts = host.split(".").map(Number)
  if (parts.length !== 4 || parts.some((p) => Number.isNaN(p))) return false
  const [a, b] = parts
  if (a === 10) return true
  if (a === 172 && b >= 16 && b <= 31) return true
  if (a === 192 && b === 168) return true
  if (a === 169 && b === 254) return true
  return false
}

function assertUrlAllowed(rawUrl: string): URL {
  let url: URL
  try {
    url = new URL(rawUrl)
  } catch {
    throw new Error(`Invalid URL: ${rawUrl}`)
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error(`Unsupported protocol: ${url.protocol} (only http/https allowed)`)
  }
  const hostname = url.hostname.toLowerCase()
  if (BLOCKED_HOSTS.has(hostname) || isPrivateIp(hostname)) {
    throw new Error(`Refusing to fetch internal/private address: ${hostname}`)
  }
  return url
}

interface FetchItArgs {
  url: string
  method?: string
  headers?: Record<string, string>
  body?: string
}

interface FetchItResult {
  status: number
  statusText: string
  headers: Record<string, string>
  cookies: string[]
  body: string
  bodyTruncated: boolean
  [key: string]: unknown
}

async function fetchIt({ url, method = "GET", headers = {}, body }: FetchItArgs): Promise<FetchItResult> {
  const target = assertUrlAllowed(url)

  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)

  try {
    const res = await fetch(target, {
      method,
      headers,
      body: body !== undefined && method !== "GET" && method !== "HEAD" ? body : undefined,
      signal: controller.signal,
      redirect: "follow",
    })

    const responseHeaders: Record<string, string> = {}
    res.headers.forEach((value, key) => {
      if (key.toLowerCase() !== "set-cookie") responseHeaders[key] = value
    })

    const cookies: string[] =
      typeof (res.headers as unknown as { getSetCookie?: () => string[] }).getSetCookie === "function"
        ? (res.headers as unknown as { getSetCookie: () => string[] }).getSetCookie()
        : []

    const rawBody = await res.text()
    const truncated = rawBody.length > MAX_BODY_CHARS
    const bodyOut = truncated ? rawBody.slice(0, MAX_BODY_CHARS) : rawBody

    return {
      status: res.status,
      statusText: res.statusText,
      headers: responseHeaders,
      cookies,
      body: bodyOut,
      bodyTruncated: truncated,
    }
  } catch (err) {
    if (err instanceof Error && err.name === "AbortError") {
      throw new Error(`Request timed out after ${REQUEST_TIMEOUT_MS}ms`)
    }
    throw err
  } finally {
    clearTimeout(timeout)
  }
}

// ============================================================================
// MCP Server Setup
// ============================================================================

// Fresh server per request — stateless streamable-HTTP mode only allows one
// transport per Server, and Cloud Run opens several connections (startup
// probe + real requests), so a module-scope singleton would throw
// "Already connected to a transport".
function createMcpServer(): McpServer {
  const server = new McpServer({
    name: "fetchit",
    version: "1.0.0",
  })

  server.registerTool(
    "fetch_it",
    {
      title: "FetchIt",
      description:
        "Make a real HTTP request to any public URL and return the raw response: " +
        "status code, response headers, cookies (Set-Cookie), and body. Use this " +
        "to test/inspect an actual API's response instead of guessing — for " +
        "example checking auth error formats, response shapes, rate-limit headers, " +
        "or cookies a login endpoint sets. Only http/https URLs to public hosts " +
        "are allowed; internal/private/localhost addresses are blocked.",
      inputSchema: {
        url: z.string().url().describe("Full URL to request, e.g. https://api.example.com/users"),
        method: z.string().optional().describe("HTTP method (default: GET)"),
        headers: z.record(z.string(), z.string()).optional().describe("Request headers as key/value pairs"),
        body: z.string().optional().describe("Raw request body (for POST/PUT/PATCH etc.)"),
      },
      outputSchema: {
        status: z.number(),
        statusText: z.string(),
        headers: z.record(z.string(), z.string()),
        cookies: z.array(z.string()),
        body: z.string(),
        bodyTruncated: z.boolean(),
      },
    },
    async ({ url, method, headers, body }) => {
      const output = await fetchIt({ url, method, headers, body })
      return {
        content: [{ type: "text", text: JSON.stringify(output) }],
        structuredContent: output,
      }
    }
  )

  return server
}

// ============================================================================
// Express App Setup
// ============================================================================

const app = express()
app.use(express.json())
app.get("/health", (_req: Request, res: Response) => res.status(200).json({ status: "healthy" }))
app.post("/mcp", async (req: Request, res: Response) => {
  const startTime = Date.now()
  const body = req.body
  const method = body?.method || "unknown"
  const params = body?.params
  if (method === "tools/call") {
    const toolName = params?.name || "unknown"
    const toolArgs = params?.arguments
    logRequest(`tools/call ${chalk.bold(toolName)}`, toolArgs)
  }
  else if (method !== "notifications/initialized") logRequest(method, params)
  const transport = new StreamableHTTPServerTransport({
    sessionIdGenerator: undefined,
    enableJsonResponse: true
  })
  let responseBody = ""
  const originalWrite = res.write.bind(res) as typeof res.write
  const originalEnd = res.end.bind(res) as typeof res.end
  res.write = function (chunk: unknown, encodingOrCallback?: BufferEncoding | ((error: Error | null | undefined) => void), callback?: (error: Error | null | undefined) => void) {
    if (chunk) responseBody += typeof chunk === "string" ? chunk : Buffer.from(chunk as ArrayBuffer).toString()
    return originalWrite(chunk as string, encodingOrCallback as BufferEncoding, callback)
  }
  res.end = function (chunk?: unknown, encodingOrCallback?: BufferEncoding | (() => void), callback?: () => void) {
    if (chunk) responseBody += typeof chunk === "string" ? chunk : Buffer.from(chunk as ArrayBuffer).toString()
    if (method !== "notifications/initialized") {
      const latency = Date.now() - startTime
      try {
        const rpcResponse = JSON.parse(responseBody) as { result?: unknown; error?: unknown }
        if (rpcResponse?.error) logError(method, rpcResponse.error, latency)
        else if (method === "tools/call") {
          const content = (rpcResponse?.result as { content?: Array<{ text?: string }> })?.content
          const resultText = content?.[0]?.text
          logResponse(method, resultText, latency)
        }
        else logResponse(method, null, latency)
      } catch { logResponse(method, null, latency) }
    }
    return originalEnd(chunk as string, encodingOrCallback as BufferEncoding, callback)
  }
  res.on("close", () => transport.close())
  const server = createMcpServer()
  await server.connect(transport)
  await transport.handleRequest(req, res, req.body)
})
app.use((_err: unknown, _req: Request, res: Response, _next: Function) => res.status(500).json({ error: "Internal server error" }))

// ============================================================================
// Start Server
// ============================================================================

const port = parseInt(process.env.PORT || "8080")
const httpServer = app.listen(port, () => {
  console.log()
  console.log(chalk.bold("MCP Server running on"), chalk.cyan(`http://localhost:${port}`))
  console.log(`  ${chalk.gray("Health:")} http://localhost:${port}/health`)
  console.log(`  ${chalk.gray("MCP:")}    http://localhost:${port}/mcp`)
  if (isDev) {
    console.log()
    console.log(chalk.gray("─".repeat(50)))
    console.log()
  }
})

process.on("SIGTERM", () => {
  console.log("Received SIGTERM, shutting down...")
  httpServer.close(() => process.exit(0))
})
