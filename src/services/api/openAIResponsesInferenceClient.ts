import type {
  BetaMessage,
  BetaRawMessageStreamEvent,
} from '@anthropic-ai/sdk/resources/beta/messages/messages.mjs'
import { Stream as SDKStream } from '@anthropic-ai/sdk/streaming.mjs'
import { createHash, randomUUID } from 'crypto'
import { parseSSEFrames } from '../../cli/transports/SSETransport.js'
import { TOOL_SEARCH_TOOL_NAME } from '../../tools/ToolSearchTool/constants.js'
import { errorMessage } from '../../utils/errors.js'
import { modelSupportsOpenAIResponsesToolSearch } from '../../utils/model/providers.js'
import type {
  InferenceClient,
  InferenceCountTokensArgs,
  InferenceCountTokensResult,
  InferenceCreateMessageArgs,
  InferenceCreateMessageResult,
  InferenceListModelsArgs,
  InferenceListModelsResult,
} from './inferenceClient.js'
import {
  OpenAICompatHTTPError,
  OpenAICompatTransportError,
} from './openAICompatInferenceClient.js'

export const OPENAI_RESPONSE_ITEMS_FIELD = '_openai_response_items'

type FetchLike = typeof fetch
type ResponseItem = Record<string, unknown> & { type: string }
type ResponseUsage = {
  input_tokens?: number
  output_tokens?: number
  input_tokens_details?: { cached_tokens?: number } | null
}
type OpenAIResponse = {
  id?: string
  model?: string
  status?: string
  incomplete_details?: { reason?: string } | null
  error?: { code?: string; message?: string; type?: string } | null
  output?: ResponseItem[]
  usage?: ResponseUsage | null
}

type ResponsesStopReason = 'tool_use' | 'max_tokens' | 'end_turn' | 'refusal'

type Options = {
  baseURL: string
  fetch?: FetchLike
  headers?: HeadersInit
  useNCodeManagedModelRouting?: boolean
  wsV2Transport?: null
}

function usage(value?: ResponseUsage | null) {
  const input = value?.input_tokens ?? 0
  const cached = Math.min(
    input,
    Math.max(value?.input_tokens_details?.cached_tokens ?? 0, 0),
  )
  return {
    input_tokens: input - cached,
    output_tokens: value?.output_tokens ?? 0,
    cache_creation_input_tokens: 0,
    cache_read_input_tokens: cached,
    server_tool_use: { web_search_requests: 0, web_fetch_requests: 0 },
  }
}

function systemText(system: unknown): string | undefined {
  if (typeof system === 'string') return system || undefined
  if (!Array.isArray(system)) return undefined
  const text = system
    .filter(
      (block): block is { type: 'text'; text: string } =>
        !!block &&
        typeof block === 'object' &&
        block.type === 'text' &&
        typeof block.text === 'string',
    )
    .map(block => block.text)
    .join('')
  return text || undefined
}

function contentText(content: unknown): string {
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return String(content ?? '')
  return content
    .map(block => {
      if (!block || typeof block !== 'object') return ''
      if ('text' in block && typeof block.text === 'string') return block.text
      return ''
    })
    .join('')
}

function responseItemKey(item: ResponseItem): string {
  if (typeof item.id === 'string') return `${item.type}:id:${item.id}`
  if (typeof item.call_id === 'string') {
    return `${item.type}:call_id:${item.call_id}`
  }
  return `${item.type}:value:${JSON.stringify(item)}`
}

function parseToolArguments(value: unknown): unknown {
  if (typeof value !== 'string') return value ?? {}
  try {
    return JSON.parse(value)
  } catch {
    return value
  }
}

function toolCallID(item: ResponseItem): string {
  const callID = item.call_id ?? item.id
  if (typeof callID !== 'string' || callID.length === 0) {
    throw new OpenAICompatTransportError(
      `Responses API ${item.type} item is missing a call_id`,
    )
  }
  return callID
}

function convertInputBlock(block: unknown): Record<string, unknown> {
  if (!block || typeof block !== 'object' || !('type' in block)) {
    throw new Error('Unsupported empty Responses input content block')
  }
  if (
    block.type === 'text' &&
    'text' in block &&
    typeof block.text === 'string'
  ) {
    return { type: 'input_text', text: block.text }
  }
  if (
    block.type === 'image' &&
    'source' in block &&
    block.source &&
    typeof block.source === 'object'
  ) {
    const source = block.source as Record<string, unknown>
    if (
      source.type === 'base64' &&
      typeof source.data === 'string' &&
      typeof source.media_type === 'string'
    ) {
      return {
        type: 'input_image',
        image_url: `data:${source.media_type};base64,${source.data}`,
      }
    }
    if (source.type === 'url' && typeof source.url === 'string') {
      return { type: 'input_image', image_url: source.url }
    }
    if (source.type === 'file' && typeof source.file_id === 'string') {
      return { type: 'input_image', file_id: source.file_id }
    }
    throw new Error(
      `Unsupported Responses image source: ${String(source.type)}`,
    )
  }
  if (
    block.type === 'document' &&
    'source' in block &&
    block.source &&
    typeof block.source === 'object'
  ) {
    const source = block.source as Record<string, unknown>
    if (
      source.type === 'base64' &&
      typeof source.data === 'string' &&
      typeof source.media_type === 'string'
    ) {
      return {
        type: 'input_file',
        filename:
          'title' in block && typeof block.title === 'string'
            ? block.title
            : 'document.pdf',
        file_data: `data:${source.media_type};base64,${source.data}`,
      }
    }
    if (source.type === 'url' && typeof source.url === 'string') {
      return { type: 'input_file', file_url: source.url }
    }
    if (source.type === 'file' && typeof source.file_id === 'string') {
      return { type: 'input_file', file_id: source.file_id }
    }
    if (source.type === 'text' && typeof source.data === 'string') {
      return { type: 'input_text', text: source.data }
    }
    throw new Error(
      `Unsupported Responses document source: ${String(source.type)}`,
    )
  }
  throw new Error(
    `Unsupported Responses input content block: ${String(block.type)}`,
  )
}

function convertInputContent(
  content: unknown,
): string | Record<string, unknown>[] {
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) {
    throw new Error('Responses input content must be a string or array')
  }
  return content.map(convertInputBlock)
}

function convertFunctionTool(
  tool: Record<string, unknown>,
  deferLoading = tool.defer_loading === true,
): ResponseItem {
  return {
    type: 'function',
    name: tool.name,
    description: tool.description,
    parameters: tool.input_schema ?? { type: 'object', properties: {} },
    ...(tool.strict !== undefined ? { strict: tool.strict } : {}),
    ...(deferLoading ? { defer_loading: true } : {}),
  }
}

function toolMap(tools: unknown): Map<string, Record<string, unknown>> {
  if (!Array.isArray(tools)) return new Map()
  return new Map(
    tools
      .filter(
        (tool): tool is Record<string, unknown> =>
          !!tool &&
          typeof tool === 'object' &&
          typeof (tool as Record<string, unknown>).name === 'string',
      )
      .map(tool => [String(tool.name), tool]),
  )
}

function toolSearchOutput(
  callID: string,
  names: string[],
  toolsByName: Map<string, Record<string, unknown>>,
): ResponseItem {
  const selectedTools = names.map(name => {
    const tool = toolsByName.get(name)
    if (!tool) {
      throw new OpenAICompatTransportError(
        `Tool search selected unavailable tool: ${name}`,
      )
    }
    return convertFunctionTool(tool, true)
  })
  return {
    type: 'tool_search_output',
    call_id: callID,
    status: 'completed',
    execution: 'client',
    tools: selectedTools,
  }
}

function toolReferenceNames(content: unknown): string[] {
  if (!Array.isArray(content)) return []
  const names = content
    .filter(
      (block): block is { type: 'tool_reference'; tool_name: string } =>
        !!block &&
        typeof block === 'object' &&
        block.type === 'tool_reference' &&
        typeof block.tool_name === 'string',
    )
    .map(block => block.tool_name)
  return [...new Set(names)]
}

function functionCallOutput(content: unknown): string | Record<string, unknown>[] {
  if (!Array.isArray(content)) return convertInputContent(content)
  const blocks = content.filter(
    block =>
      !(
        block &&
        typeof block === 'object' &&
        block.type === 'tool_reference'
      ),
  )
  if (blocks.length > 0) return convertInputContent(blocks)
  const names = toolReferenceNames(content)
  return names.length > 0 ? `Loaded tools: ${names.join(', ')}` : ''
}

function toolSearchCallID(callID: string, names: string[]): string {
  const digest = createHash('sha256')
    .update(`${callID}:${names.join(',')}`)
    .digest('hex')
    .slice(0, 24)
  return `ncode_tool_load_${digest}`
}

function convertMessages(
  messages: unknown,
  tools: unknown,
  model: string,
): ResponseItem[] {
  if (!Array.isArray(messages)) {
    throw new Error('Inference messages must be an array')
  }
  const input: ResponseItem[] = []
  const seenResponseItems = new Set<string>()
  const loadedToolNames = new Set<string>()
  const toolsByName = toolMap(tools)
  const supportsToolSearch = modelSupportsOpenAIResponsesToolSearch(model)
  for (const message of messages) {
    if (!message || typeof message !== 'object' || !('role' in message)) continue
    const raw = (message as Record<string, unknown>)[OPENAI_RESPONSE_ITEMS_FIELD]
    if (Array.isArray(raw)) {
      for (const item of raw as ResponseItem[]) {
        if (
          !supportsToolSearch &&
          (item.type === 'tool_search_call' ||
            item.type === 'tool_search_output')
        ) {
          continue
        }
        const key = responseItemKey(item)
        if (!seenResponseItems.has(key)) {
          seenResponseItems.add(key)
          input.push(item)
          if (item.type === 'tool_search_output' && Array.isArray(item.tools)) {
            for (const tool of item.tools as ResponseItem[]) {
              if (typeof tool.name === 'string') loadedToolNames.add(tool.name)
            }
          }
        }
      }
      continue
    }
    const role = message.role
    const content = 'content' in message ? message.content : ''
    if (role === 'user') {
      if (Array.isArray(content)) {
        const toolResults = content.filter(
          block =>
            block &&
            typeof block === 'object' &&
            block.type === 'tool_result',
        ) as Array<Record<string, unknown>>
        for (const block of toolResults) {
          const callID = String(block.tool_use_id ?? '')
          input.push({
            type: 'function_call_output',
            call_id: block.tool_use_id,
            output: functionCallOutput(block.content),
          })
          const names = toolReferenceNames(block.content).filter(
            name => !loadedToolNames.has(name),
          )
          if (supportsToolSearch && names.length > 0) {
            const searchCallID = toolSearchCallID(callID, names)
            input.push({
              type: 'tool_search_call',
              call_id: searchCallID,
              status: 'completed',
              execution: 'client',
              arguments: { query: names.join(' '), limit: names.length },
            })
            input.push(toolSearchOutput(searchCallID, names, toolsByName))
            for (const name of names) loadedToolNames.add(name)
          }
        }
        const userContent = content.filter(
          block =>
            !(
              block &&
              typeof block === 'object' &&
              block.type === 'tool_result'
            ),
        )
        if (userContent.length > 0) {
          input.push({
            type: 'message',
            role: 'user',
            content: convertInputContent(userContent),
          })
        }
      } else {
        input.push({
          type: 'message',
          role: 'user',
          content: convertInputContent(content),
        })
      }
      continue
    }
    if (role === 'assistant' && Array.isArray(content)) {
      const text = contentText(content)
      if (text) input.push({ type: 'message', role: 'assistant', content: text })
      for (const block of content) {
        if (block && typeof block === 'object' && block.type === 'tool_use') {
          input.push({
            type: 'function_call',
            call_id: block.id,
            name: block.name,
            arguments: JSON.stringify(block.input ?? {}),
          })
        }
      }
    }
  }
  return input
}

function convertToolChoice(value: unknown): unknown {
  if (!value || typeof value === 'string') return value ?? 'auto'
  if (typeof value !== 'object') return 'auto'
  if ('type' in value && value.type === 'any') return 'required'
  if ('type' in value && value.type === 'none') return 'none'
  if (
    'type' in value &&
    value.type === 'tool' &&
    'name' in value &&
    typeof value.name === 'string'
  ) {
    return { type: 'function', name: value.name }
  }
  return 'auto'
}

function convertParallelToolCalls(value: unknown): boolean | undefined {
  if (!value || typeof value !== 'object') return undefined
  return 'disable_parallel_tool_use' in value &&
    value.disable_parallel_tool_use === true
    ? false
    : undefined
}

function convertOutputFormat(
  value: unknown,
): Record<string, unknown> | undefined {
  if (!value || typeof value !== 'object' || !('format' in value)) {
    return undefined
  }
  const format = value.format
  if (!format || typeof format !== 'object' || !('type' in format)) {
    return undefined
  }
  if (format.type === 'json_object') return { type: 'json_object' }
  if (format.type === 'json_schema' && 'schema' in format) {
    const schema = format.schema
    return {
      type: 'json_schema',
      name:
        'name' in format && typeof format.name === 'string'
          ? format.name
          : schema &&
              typeof schema === 'object' &&
              'title' in schema &&
              typeof schema.title === 'string'
            ? schema.title
            : 'Schema',
      schema,
      strict: 'strict' in format && format.strict === true,
    }
  }
  throw new Error(
    `Unsupported Responses output format: ${String(format.type)}`,
  )
}

function convertTools(tools: unknown): ResponseItem[] | undefined {
  if (!Array.isArray(tools) || tools.length === 0) return undefined
  const records = tools as Array<Record<string, unknown>>
  const searchTool = records.find(tool => tool.name === TOOL_SEARCH_TOOL_NAME)
  const hasDeferredTools = records.some(tool => tool.defer_loading === true)
  if (hasDeferredTools && !searchTool) {
    throw new OpenAICompatTransportError(
      'Deferred Responses tools require the NCode ToolSearch tool',
    )
  }
  const converted = records
    .filter(tool => tool.defer_loading !== true)
    .map(tool => convertFunctionTool(tool, false))
  return converted.length > 0 ? converted : undefined
}

export function buildOpenAIResponsesRequest(
  params: InferenceCreateMessageArgs[0],
) {
  const reasoning =
    params.thinking?.type === 'disabled'
      ? { effort: 'none' }
      : params.thinking
        ? {
            effort:
              (params.output_config as { effort?: string } | undefined)
                ?.effort ?? 'high',
            summary: 'auto',
          }
        : undefined
  const outputFormat = convertOutputFormat(params.output_config)
  const parallelToolCalls = convertParallelToolCalls(params.tool_choice)
  return {
    model: params.model,
    input: convertMessages(params.messages, params.tools, params.model),
    ...(systemText(params.system)
      ? { instructions: systemText(params.system) }
      : {}),
    ...(convertTools(params.tools) ? { tools: convertTools(params.tools) } : {}),
    tool_choice: convertToolChoice(params.tool_choice),
    ...(parallelToolCalls !== undefined
      ? { parallel_tool_calls: parallelToolCalls }
      : {}),
    max_output_tokens: params.max_tokens,
    ...(reasoning ? { reasoning } : {}),
    ...(outputFormat ? { text: { format: outputFormat } } : {}),
    ...(reasoning && reasoning.effort !== 'none'
      ? { include: ['reasoning.encrypted_content'] }
      : {}),
    store: false,
    ...(params.stream ? { stream: true } : {}),
  }
}

function incompleteStopReason(
  response: OpenAIResponse,
): Extract<ResponsesStopReason, 'max_tokens' | 'refusal'> | undefined {
  if (response.status !== 'incomplete') return undefined
  switch (response.incomplete_details?.reason) {
    case 'max_output_tokens':
      return 'max_tokens'
    case 'content_filter':
      return 'refusal'
    default:
      throw new OpenAICompatTransportError(
        `Responses API returned an incomplete response without a supported reason: ${String(response.incomplete_details?.reason)}`,
      )
  }
}

function responseStatusDetail(response: OpenAIResponse): string {
  const code = response.error?.code?.trim()
  const type = response.error?.type?.trim()
  const message = response.error?.message?.replace(/\s+/g, ' ').trim()
  const label = [...new Set([code, type].filter(Boolean))].join('/')
  return [label, message].filter(Boolean).join(': ').slice(0, 1000)
}

function assertTerminalResponse(response: OpenAIResponse): void {
  if (response.status === 'completed' || response.status === 'incomplete') {
    return
  }
  const detail = responseStatusDetail(response)
  if (response.status === 'failed') {
    throw new OpenAICompatTransportError(
      `Responses API response failed${detail ? `: ${detail}` : ''}`,
      response.error,
    )
  }
  if (response.status === 'cancelled') {
    throw new Error(
      `Responses API response was cancelled${detail ? `: ${detail}` : ''}`,
    )
  }
  throw new OpenAICompatTransportError(
    `Responses API returned non-terminal status: ${String(response.status)}`,
  )
}

function refusalText(response: OpenAIResponse): string {
  return (response.output ?? [])
    .flatMap(item =>
      item.type === 'message' && Array.isArray(item.content)
        ? (item.content as Array<Record<string, unknown>>)
        : [],
    )
    .filter(part => part.type === 'refusal' && typeof part.refusal === 'string')
    .map(part => String(part.refusal))
    .join('')
}

function messageFromResponse(response: OpenAIResponse): BetaMessage {
  assertTerminalResponse(response)
  const content: Array<Record<string, unknown>> = []
  const refusal = refusalText(response)
  const incompleteReason = incompleteStopReason(response)
  for (const item of response.output ?? []) {
    if (item.type === 'message' && Array.isArray(item.content)) {
      for (const part of item.content as Array<Record<string, unknown>>) {
        if (part.type === 'output_text' && typeof part.text === 'string') {
          content.push({ type: 'text', text: part.text })
        } else if (part.type === 'refusal' && typeof part.refusal === 'string') {
          content.push({ type: 'text', text: part.refusal })
        }
      }
    } else if (item.type === 'function_call') {
      content.push({
        type: 'tool_use',
        id: toolCallID(item),
        name: String(item.name ?? ''),
        input: parseToolArguments(item.arguments),
      })
    } else if (item.type === 'reasoning' && Array.isArray(item.summary)) {
      const text = (item.summary as Array<Record<string, unknown>>)
        .map(part => (typeof part.text === 'string' ? part.text : ''))
        .join('')
      if (text) content.push({ type: 'thinking', thinking: text, signature: '' })
    }
  }
  return {
    id: response.id ?? randomUUID(),
    type: 'message',
    role: 'assistant',
    model: response.model ?? 'unknown',
    content,
    stop_reason: refusal
      ? 'refusal'
      : content.some(block => block.type === 'tool_use')
      ? 'tool_use'
      : (incompleteReason ?? 'end_turn'),
    stop_sequence: null,
    usage: usage(response.usage),
    [OPENAI_RESPONSE_ITEMS_FIELD]: response.output ?? [],
  } as unknown as BetaMessage
}

async function* streamEvents(
  response: Response,
  fallbackModel: string,
  controller: AbortController,
): AsyncGenerator<BetaRawMessageStreamEvent> {
  if (!response.body) throw new Error('Responses stream body missing')
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let id = randomUUID()
  let model = fallbackModel
  let nextIndex = 0
  let textIndex: number | undefined
  let thinkingIndex: number | undefined
  const tools = new Map<number, { index: number; id: string; name: string }>()
  const items: ResponseItem[] = []
  let finalUsage = usage()
  let stopReason: ResponsesStopReason = 'end_turn'
  let terminalReceived = false

  yield {
    type: 'message_start',
    message: {
      id,
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: finalUsage,
      [OPENAI_RESPONSE_ITEMS_FIELD]: items,
    },
  } as unknown as BetaRawMessageStreamEvent

  try {
    streamLoop: while (true) {
      const { done, value } = await reader.read()
      buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done })
      const parsed = parseSSEFrames(buffer)
      buffer = parsed.remaining
      for (const frame of parsed.frames) {
        if (!frame.data || frame.data === '[DONE]') continue
        const event = JSON.parse(frame.data) as Record<string, unknown>
        const type = String(event.type ?? '')
        if (type === 'response.created') {
          const r = event.response as OpenAIResponse
          id = r.id ?? id
          model = r.model ?? model
        } else if (type === 'response.output_text.delta') {
          if (textIndex === undefined) {
            textIndex = nextIndex++
            yield {
              type: 'content_block_start',
              index: textIndex,
              content_block: { type: 'text', text: '' },
            } as BetaRawMessageStreamEvent
          }
          yield {
            type: 'content_block_delta',
            index: textIndex,
            delta: { type: 'text_delta', text: String(event.delta ?? '') },
          } as BetaRawMessageStreamEvent
        } else if (type === 'response.refusal.delta') {
          stopReason = 'refusal'
          if (textIndex === undefined) {
            textIndex = nextIndex++
            yield {
              type: 'content_block_start',
              index: textIndex,
              content_block: { type: 'text', text: '' },
            } as BetaRawMessageStreamEvent
          }
          yield {
            type: 'content_block_delta',
            index: textIndex,
            delta: { type: 'text_delta', text: String(event.delta ?? '') },
          } as BetaRawMessageStreamEvent
        } else if (type === 'response.reasoning_summary_text.delta') {
          if (thinkingIndex === undefined) {
            thinkingIndex = nextIndex++
            yield {
              type: 'content_block_start',
              index: thinkingIndex,
              content_block: { type: 'thinking', thinking: '', signature: '' },
            } as BetaRawMessageStreamEvent
          }
          yield {
            type: 'content_block_delta',
            index: thinkingIndex,
            delta: {
              type: 'thinking_delta',
              thinking: String(event.delta ?? ''),
            },
          } as BetaRawMessageStreamEvent
        } else if (type === 'response.output_item.added') {
          const item = event.item as ResponseItem
          if (item?.type === 'function_call') {
            const outputIndex = Number(event.output_index ?? tools.size)
            const state = {
              index: nextIndex++,
              id: String(item.call_id ?? item.id ?? randomUUID()),
              name: String(item.name ?? ''),
            }
            tools.set(outputIndex, state)
            stopReason = 'tool_use'
            yield {
              type: 'content_block_start',
              index: state.index,
              content_block: {
                type: 'tool_use',
                id: state.id,
                name: state.name,
                input: '',
              },
            } as BetaRawMessageStreamEvent
          }
        } else if (type === 'response.function_call_arguments.delta') {
          const state = tools.get(Number(event.output_index ?? 0))
          if (state) {
            yield {
              type: 'content_block_delta',
              index: state.index,
              delta: {
                type: 'input_json_delta',
                partial_json: String(event.delta ?? ''),
              },
            } as BetaRawMessageStreamEvent
          }
        } else if (type === 'response.output_item.done') {
          const outputIndex = Number(event.output_index ?? items.length)
          const item = event.item as ResponseItem
          items[outputIndex] = item
        } else if (
          type === 'response.completed' ||
          type === 'response.incomplete' ||
          type === 'response.failed' ||
          type === 'response.cancelled'
        ) {
          const r = event.response as OpenAIResponse
          terminalReceived = true
          assertTerminalResponse(r)
          Object.assign(finalUsage, usage(r.usage))
          if (r.output?.length) {
            items.splice(0, items.length, ...r.output)
          }
          const refusal = refusalText(r)
          if (refusal) {
            stopReason = 'refusal'
            if (textIndex === undefined) {
              textIndex = nextIndex++
              yield {
                type: 'content_block_start',
                index: textIndex,
                content_block: { type: 'text', text: '' },
              } as BetaRawMessageStreamEvent
              yield {
                type: 'content_block_delta',
                index: textIndex,
                delta: { type: 'text_delta', text: refusal },
              } as BetaRawMessageStreamEvent
            }
          } else stopReason = incompleteStopReason(r) ?? stopReason
          break streamLoop
        } else if (type === 'error') {
          terminalReceived = true
          throw new OpenAICompatTransportError(
            `Responses API stream failed: ${JSON.stringify(event.error ?? event)}`,
            event.error,
          )
        }
      }
      if (done) break
    }
  } finally {
    await reader.cancel().catch(() => {})
    reader.releaseLock()
  }
  if (!terminalReceived) {
    throw new OpenAICompatTransportError(
      'Responses API stream ended before a terminal response event',
    )
  }
  for (const state of tools.values()) {
    yield {
      type: 'content_block_stop',
      index: state.index,
    } as BetaRawMessageStreamEvent
  }
  if (thinkingIndex !== undefined) {
    yield {
      type: 'content_block_stop',
      index: thinkingIndex,
    } as BetaRawMessageStreamEvent
  }
  if (textIndex !== undefined) {
    yield {
      type: 'content_block_stop',
      index: textIndex,
    } as BetaRawMessageStreamEvent
  }
  yield {
    type: 'message_delta',
    delta: { stop_reason: stopReason, stop_sequence: null },
    usage: finalUsage,
  } as BetaRawMessageStreamEvent
  yield { type: 'message_stop' } as BetaRawMessageStreamEvent
}

function operation<T>(
  responsePromise: Promise<Response>,
  parse: (r: Response) => Promise<T>,
) {
  const parsed = responsePromise.then(parse)
  return Object.assign(parsed, {
    asResponse: () => responsePromise,
    withResponse: async () => {
      const [data, response] = await Promise.all([parsed, responsePromise])
      return { data, response, request_id: response.headers.get('x-request-id') }
    },
  })
}

async function responseError(response: Response): Promise<OpenAICompatHTTPError> {
  let detail: string | undefined
  let errorCode: string | undefined
  let errorType: string | undefined
  try {
    const body = (await response.json()) as Record<string, unknown>
    const nested =
      body.error && typeof body.error === 'object'
        ? (body.error as Record<string, unknown>)
        : undefined
    const candidate = nested?.message ?? body.message ?? body.detail
    if (typeof candidate === 'string' && candidate.trim()) {
      detail = candidate.replace(/\s+/g, ' ').trim().slice(0, 1000)
    }
    if (typeof nested?.code === 'string') errorCode = nested.code
    if (typeof nested?.type === 'string') errorType = nested.type
  } catch {
    // Non-JSON error pages do not contain safe, structured API diagnostics.
  }
  return new OpenAICompatHTTPError(
    response.status,
    response.statusText,
    detail,
    { headers: response.headers, errorCode, errorType },
  )
}

export class OpenAIResponsesInferenceClient implements InferenceClient {
  private readonly fetchImpl: FetchLike
  constructor(private readonly options: Options) {
    this.fetchImpl = options.fetch ?? globalThis.fetch
  }

  private url(path: string): string {
    const base = new URL(this.options.baseURL)
    const basePath = base.pathname.replace(/\/+$/, '')
    const requestPath = path.replace(/^\/+/, '')
    const relativePath =
      basePath.endsWith('/v1') && requestPath.startsWith('v1/')
        ? requestPath.slice(3)
        : requestPath
    base.pathname = [basePath, relativePath]
      .filter(Boolean)
      .join('/')
      .replace(/\/{2,}/g, '/')
    return base.toString()
  }

  private request(
    path: string,
    body: unknown,
    signal?: AbortSignal,
    extra?: HeadersInit,
  ) {
    const headers = new Headers(this.options.headers)
    headers.set('content-type', 'application/json')
    headers.set('accept', 'application/json')
    for (const [key, value] of new Headers(extra).entries()) {
      headers.set(key, value)
    }
    return this.fetchImpl(this.url(path), {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal,
    }).catch(error => {
      if (signal?.aborted) throw error
      throw new OpenAICompatTransportError(errorMessage(error), error)
    })
  }

  createMessage(...args: InferenceCreateMessageArgs): InferenceCreateMessageResult {
    const [params, options] = args
    const controller = new AbortController()
    if (options?.signal?.aborted) {
      controller.abort(options.signal.reason)
    } else if (options?.signal) {
      options.signal.addEventListener(
        'abort',
        () => controller.abort(options.signal?.reason),
        { once: true },
      )
    }
    const responsePromise = this.request(
      '/v1/responses',
      buildOpenAIResponsesRequest(params),
      controller.signal,
      options?.headers,
    )
    return operation(responsePromise, async response => {
      if (!response.ok) throw await responseError(response)
      if (params.stream) return new SDKStream(() => streamEvents(response, params.model, controller), controller) as never
      return messageFromResponse((await response.json()) as OpenAIResponse) as never
    }) as InferenceCreateMessageResult
  }

  countTokens(...args: InferenceCountTokensArgs): InferenceCountTokensResult {
    const [params, options] = args
    const createRequest = buildOpenAIResponsesRequest({
      ...params,
      max_tokens: 1,
      stream: false,
    } as InferenceCreateMessageArgs[0])
    const {
      include: _include,
      max_output_tokens: _maxOutputTokens,
      store: _store,
      ...request
    } = createRequest
    return this.request(
      '/v1/responses/input_tokens',
      request,
      options?.signal,
      options?.headers,
    ).then(async response => {
      if (!response.ok) {
        throw await responseError(response)
      }
      const body = (await response.json()) as { input_tokens?: number }
      if (typeof body.input_tokens !== 'number') {
        throw new OpenAICompatTransportError(
          'Responses input-token endpoint returned no input_tokens value',
        )
      }
      return { input_tokens: body.input_tokens }
    }) as InferenceCountTokensResult
  }

  listModels(...args: InferenceListModelsArgs): InferenceListModelsResult {
    const [, options] = args
    const headers = new Headers(this.options.headers)
    for (const [key, value] of new Headers(options?.headers).entries()) headers.set(key, value)
    const promise = this.fetchImpl(this.url('/v1/models'), { headers, signal: options?.signal })
    return { async *[Symbol.asyncIterator]() { const response = await promise; if (!response.ok) throw await responseError(response); const body = await response.json() as { data?: Array<Record<string, unknown>> }; for (const item of body.data ?? []) yield item } } as InferenceListModelsResult
  }
}
