import type Anthropic from '@anthropic-ai/sdk'
import {
  getAPIProvider,
  getNoumenaBaseUrl,
  getOpenAICompatBaseUrl,
  isOpenAICompatByokActive,
  isFirstPartyNoumenaBaseUrl,
} from '../../utils/model/providers.js'
import {
  getAnthropicClient,
  getFirstPartyRequestHeaders,
  getWrappedClientFetch,
} from './client.js'
import { OpenAICompatInferenceClient } from './openAICompatInferenceClient.js'
import { OpenAIResponsesInferenceClient } from './openAIResponsesInferenceClient.js'
import { getOpenAIApiFormat } from '../../utils/model/providers.js'
import { getNCodeManagedModelBaseUrl } from '../../utils/model/ncodeModels.js'
import { getDirectApiKeyEnvValue } from '../../utils/authEnv.js'
import { getUserAgent } from '../../utils/http.js'

export type InferenceCreateMessageArgs = Parameters<
  Anthropic['beta']['messages']['create']
>
export type InferenceCreateMessageResult = ReturnType<
  Anthropic['beta']['messages']['create']
>

export type InferenceCountTokensArgs = Parameters<
  Anthropic['beta']['messages']['countTokens']
>
export type InferenceCountTokensResult = ReturnType<
  Anthropic['beta']['messages']['countTokens']
>

export type InferenceListModelsArgs = Parameters<Anthropic['models']['list']>
export type InferenceListModelsResult = ReturnType<Anthropic['models']['list']>

/**
 * `code/`'s inference seam must preserve the full caller-visible information
 * set, even if Noumena later changes the transport or payload format.
 *
 * Keep the methods below aligned with what current call sites actually observe:
 * - `createMessage()` result identity plus `.withResponse()` / `.asResponse()`
 * - `countTokens()` response fields
 * - `listModels()` async iteration shape
 */
export interface InferenceClient {
  createMessage(...args: InferenceCreateMessageArgs): InferenceCreateMessageResult
  countTokens(...args: InferenceCountTokensArgs): InferenceCountTokensResult
  listModels(...args: InferenceListModelsArgs): InferenceListModelsResult
  compactResponse?(
    params: InferenceCreateMessageArgs[0],
    options?: InferenceCreateMessageArgs[1],
  ): Promise<{ output: Array<Record<string, unknown>>; usage?: unknown }>
}

export function assertCompactedStateCompatible(
  client: InferenceClient,
  messages: unknown[],
): void {
  if (client.compactResponse) return
  const hasResponsesCompaction = messages.some(message => {
    if (!message || typeof message !== 'object') return false
    const items = (message as Record<string, unknown>)._openai_response_items
    return (
      Array.isArray(items) &&
      items.some(
        item =>
          item &&
          typeof item === 'object' &&
          (item as Record<string, unknown>).type === 'compaction',
      )
    )
  })
  if (hasResponsesCompaction) {
    throw new Error(
      'This session contains OpenAI Responses compacted state that the selected API transport cannot replay. Resume it with the same OpenAI Responses provider, including OPENAI_API_FORMAT=responses, or start a new session before switching providers.',
    )
  }
}

class AnthropicInferenceClient implements InferenceClient {
  constructor(private readonly anthropic: Anthropic) {}

  private stripResponsesMetadata<T extends { messages: unknown[] }>(
    params: T,
  ): T {
    const messages = params.messages.map(message => {
      const {
        _openai_response_items: _ignored,
        ...anthropicMessage
      } = message as Record<string, unknown>
      return anthropicMessage
    })
    return { ...params, messages } as T
  }

  createMessage(
    ...args: InferenceCreateMessageArgs
  ): InferenceCreateMessageResult {
    const [params, options] = args
    return this.anthropic.beta.messages.create(
      this.stripResponsesMetadata(params),
      options,
    )
  }

  countTokens(
    ...args: InferenceCountTokensArgs
  ): InferenceCountTokensResult {
    const [params, options] = args
    return this.anthropic.beta.messages.countTokens(
      this.stripResponsesMetadata(params),
      options,
    )
  }

  listModels(...args: InferenceListModelsArgs): InferenceListModelsResult {
    return this.anthropic.models.list(...args)
  }
}

function getLegacyOpenAICompatBaseUrl(): string | undefined {
  const legacyBaseUrl = process.env.ANTHROPIC_BASE_URL?.trim()
  if (!legacyBaseUrl) {
    return undefined
  }
  if (
    isFirstPartyNoumenaBaseUrl() ||
    isZaiAnthropicMessagesBaseUrl(legacyBaseUrl)
  ) {
    return undefined
  }
  return legacyBaseUrl
}

function getZaiAnthropicMessagesBaseUrl(): string | undefined {
  const baseUrl = process.env.ANTHROPIC_BASE_URL?.trim()
  return baseUrl && isZaiAnthropicMessagesBaseUrl(baseUrl) ? baseUrl : undefined
}

function isZaiAnthropicMessagesBaseUrl(baseUrl: string): boolean {
  try {
    const url = new URL(baseUrl)
    const path = url.pathname.replace(/\/+$/, '')
    return (
      url.hostname.toLowerCase() === 'api.z.ai' && path === '/api/anthropic'
    )
  } catch {
    return false
  }
}

function getOpenAICompatByokHeaders(apiKey: string): Record<string, string> {
  return {
    Authorization: `Bearer ${apiKey}`,
    'User-Agent': getUserAgent(),
  }
}

export async function getInferenceClient(
  args: Parameters<typeof getAnthropicClient>[0],
): Promise<InferenceClient> {
  if (isOpenAICompatByokActive()) {
    const apiKey = getDirectApiKeyEnvValue()
    const baseURL = getOpenAICompatBaseUrl()
    if (apiKey && baseURL) {
      const Client =
        getOpenAIApiFormat() === 'responses'
          ? OpenAIResponsesInferenceClient
          : OpenAICompatInferenceClient
      return new Client({
        baseURL,
        headers: getOpenAICompatByokHeaders(apiKey),
        useNCodeManagedModelRouting: false,
        wsV2Transport: null,
        ...(args.fetchOverride ? { fetch: args.fetchOverride } : {}),
      })
    }
  }

  if (getAPIProvider() === 'firstParty') {
    if (getZaiAnthropicMessagesBaseUrl()) {
      return new AnthropicInferenceClient(await getAnthropicClient(args))
    }

    const managedModelBaseURL = getNCodeManagedModelBaseUrl(args.model)
    const configuredCompatBaseURL =
      getNoumenaBaseUrl() ?? getLegacyOpenAICompatBaseUrl()
    const baseURL = managedModelBaseURL ?? configuredCompatBaseURL
    if (baseURL) {
      const headers = await getFirstPartyRequestHeaders(
        args.apiKey
          ? {
              apiKey: args.apiKey,
              includeApiKeyHeader: true,
            }
          : {},
      )
      const fetch = getWrappedClientFetch(args.fetchOverride, args.source)
      return new OpenAICompatInferenceClient({
        baseURL,
        headers,
        ...(fetch ? { fetch } : {}),
      })
    }
  }

  return new AnthropicInferenceClient(await getAnthropicClient(args))
}
