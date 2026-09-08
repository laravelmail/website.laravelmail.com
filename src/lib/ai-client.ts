import type { AIProviderInterface } from "./ai-provider";
import { LocalProvider, OpenRouterProvider } from "./ai-provider";

const DEFAULT_PRIMARY_MODEL = "hf.co/laravelcompany/laravelmail:latest";
const DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free";
const DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1";

function getOpenRouterApiKey(): string | undefined {
  if (import.meta.env.SSR) {
    return import.meta.env.OPENROUTER_API_KEY?.trim();
  }
  return undefined;
}

function getOpenRouterModel(): string {
  if (import.meta.env.SSR) {
    return import.meta.env.OPENROUTER_MODEL?.trim() || DEFAULT_OPENROUTER_MODEL;
  }
  return DEFAULT_OPENROUTER_MODEL;
}

function getOpenRouterBaseUrl(): string {
  if (import.meta.env.SSR) {
    return import.meta.env.OPENROUTER_BASE_URL?.trim() || DEFAULT_OPENROUTER_BASE_URL;
  }
  return DEFAULT_OPENROUTER_BASE_URL;
}

function getPrimaryEndpoint(): string {
  const publicEndpoint = import.meta.env.PUBLIC_PRIMARY_AI_ENDPOINT?.trim();
  if (publicEndpoint) return publicEndpoint;
  return "https://llm.laravelmail.com";
}

function getFallbackEndpoint(): string {
  const publicEndpoint = import.meta.env.PUBLIC_FALLBACK_AI_ENDPOINT?.trim();
  if (publicEndpoint) return publicEndpoint;
  return "";
}

export async function fetchWithFallback(
  prompt: string,
  primaryEndpoint?: string,
  fallbackEndpoint?: string,
  timeoutMs = 30000
): Promise<string> {
  const primaryAbort = new AbortController();
  const primaryTimeout = setTimeout(() => primaryAbort.abort(), timeoutMs);

  try {
    const endpoint = primaryEndpoint || getPrimaryEndpoint();
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
      signal: primaryAbort.signal,
    });
    clearTimeout(primaryTimeout);

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const json = await res.json();
    const text = json.response || json.text || json.content || JSON.stringify(json);
    return text;
  } catch (primaryError) {
    clearTimeout(primaryTimeout);

    const fallbackAbort = new AbortController();
    const fallbackTimeout = setTimeout(() => fallbackAbort.abort(), timeoutMs);

    try {
      const endpoint = fallbackEndpoint || getFallbackEndpoint();
      if (!endpoint) throw new Error("No fallback endpoint configured");

      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
        signal: fallbackAbort.signal,
      });
      clearTimeout(fallbackTimeout);

      if (!res.ok) throw new Error(`Fallback HTTP ${res.status}`);

      const json = await res.json();
      const text = json.response || json.text || json.content || JSON.stringify(json);
      return text;
    } catch (fallbackError) {
      clearTimeout(fallbackTimeout);
      throw new Error(
        `AI fallback failed. Both primary and Cloudflare endpoints are unavailable.`
      );
    }
  }
}

export async function generateWithProviderFallback(
  prompt: string,
  options?: {
    temperature?: number;
    stream?: boolean;
    onChunk?: (chunk: string) => void;
  }
): Promise<{
  text: string;
  model: string;
  provider: string;
  reasoning?: string;
  reasoningDetails?: unknown;
  usage?: {
    promptTokens: number;
    completionTokens: number;
  };
  finishReason?: string;
}> {
  const primaryEndpoint = getPrimaryEndpoint();
  const openRouterApiKey = getOpenRouterApiKey();
  const openRouterModel = getOpenRouterModel();
  const openRouterBaseUrl = getOpenRouterBaseUrl();

  const localProvider = new LocalProvider(primaryEndpoint, DEFAULT_PRIMARY_MODEL);
  const openRouterProvider = openRouterApiKey
    ? new OpenRouterProvider(openRouterApiKey, openRouterModel, openRouterBaseUrl)
    : null;

  let accumulatedChunks: string[] = [];
  let accumulatedText = "";
  let onChunkCallback = options?.onChunk;

  // Step 1: Try local provider
  try {
    const result = await localProvider.generate(
      [{ role: "user", content: prompt }],
      { stream: options?.stream ?? false, temperature: options?.temperature, onChunk: (chunk: string) => {
        accumulatedChunks.push(chunk);
        accumulatedText = accumulatedChunks.join("");
        if (onChunkCallback) onChunkCallback(chunk);
      }},
    );

    return {
      text: result.text || accumulatedText,
      model: result.model,
      provider: result.provider,
      reasoning: result.reasoning,
      reasoningDetails: result.reasoningDetails,
      usage: result.usage,
      finishReason: result.finishReason,
    };
  } catch (primaryError) {
    console.debug("Primary local provider failed, attempting OpenRouter fallback:", primaryError);
  }

  // Step 2: Fall back to OpenRouter
  if (openRouterProvider) {
    try {
      const result = await openRouterProvider.generate(
        [{ role: "user", content: prompt }],
        { stream: options?.stream ?? false, temperature: options?.temperature, onChunk: (chunk: string) => {
          accumulatedChunks.push(chunk);
          accumulatedText = accumulatedChunks.join("");
          if (onChunkCallback) onChunkCallback(chunk);
        }},
      );

      return {
        text: result.text || accumulatedText,
        model: result.model,
        provider: result.provider,
        reasoning: result.reasoning,
        reasoningDetails: result.reasoningDetails,
        usage: result.usage,
        finishReason: result.finishReason,
      };
    } catch (openRouterError) {
      console.debug("OpenRouter fallback also failed:", openRouterError);
    }
  }

  throw new Error("AI generation failed. Both primary and fallback providers are unavailable.");
}