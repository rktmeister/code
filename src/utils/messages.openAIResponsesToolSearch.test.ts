import { afterEach, beforeEach, describe, expect, it } from 'bun:test'
import { normalizeMessagesForAPI } from './messages.js'

const originalEnv = {
  NOUMENA_API_KEY: process.env.NOUMENA_API_KEY,
  ANTHROPIC_API_KEY: process.env.ANTHROPIC_API_KEY,
  OPENAI_API_KEY: process.env.OPENAI_API_KEY,
  OPENAI_API_FORMAT: process.env.OPENAI_API_FORMAT,
}

function restoreEnv() {
  for (const [key, value] of Object.entries(originalEnv)) {
    if (value === undefined) delete process.env[key]
    else process.env[key] = value
  }
}

beforeEach(() => {
  delete process.env.NOUMENA_API_KEY
  delete process.env.ANTHROPIC_API_KEY
  process.env.OPENAI_API_KEY = 'test-key'
  process.env.OPENAI_API_FORMAT = 'responses'
})

afterEach(restoreEnv)

describe('Responses tool-search message normalization', () => {
  it('preserves tool references without Anthropic turn-boundary text', () => {
    const normalized = normalizeMessagesForAPI(
      [
        {
          type: 'assistant',
          uuid: 'assistant-1',
          timestamp: '2026-01-01T00:00:00.000Z',
          message: {
            id: 'response-1',
            type: 'message',
            role: 'assistant',
            model: 'gpt-test',
            stop_reason: 'tool_use',
            stop_sequence: null,
            usage: {
              input_tokens: 1,
              output_tokens: 1,
              cache_creation_input_tokens: 0,
              cache_read_input_tokens: 0,
              server_tool_use: {
                web_search_requests: 0,
                web_fetch_requests: 0,
              },
            },
            content: [
              {
                type: 'tool_use',
                id: 'search-1',
                name: 'ToolSearch',
                input: { query: 'repository' },
              },
            ],
          },
        },
        {
          type: 'user',
          uuid: 'user-1',
          timestamp: '2026-01-01T00:00:01.000Z',
          message: {
            role: 'user',
            content: [
              {
                type: 'tool_result',
                tool_use_id: 'search-1',
                content: [
                  {
                    type: 'tool_reference',
                    tool_name: 'RepositorySearch',
                  },
                ],
              },
            ],
          },
        },
      ] as never,
      [{ name: 'RepositorySearch' }] as never,
    )

    const user = normalized.find(message => message.type === 'user')
    expect(user).toBeDefined()
    expect(user?.message.content).toEqual([
      {
        type: 'tool_result',
        tool_use_id: 'search-1',
        content: [
          {
            type: 'tool_reference',
            tool_name: 'RepositorySearch',
          },
        ],
      },
    ])
  })

  it('keeps opaque response items when streamed assistant blocks merge', () => {
    const responseItems = [
      {
        type: 'tool_search_call',
        call_id: 'search-1',
        execution: 'client',
        status: 'completed',
        arguments: { query: 'repository' },
      },
    ]
    const assistantMessage = {
      id: 'response-1',
      type: 'message',
      role: 'assistant',
      model: 'gpt-test',
      stop_reason: 'tool_use',
      stop_sequence: null,
      usage: {
        input_tokens: 1,
        output_tokens: 1,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
        server_tool_use: {
          web_search_requests: 0,
          web_fetch_requests: 0,
        },
      },
    }
    const normalized = normalizeMessagesForAPI([
      {
        type: 'assistant',
        uuid: 'assistant-text',
        timestamp: '2026-01-01T00:00:00.000Z',
        message: {
          ...assistantMessage,
          content: [{ type: 'text', text: 'Searching.' }],
          _openai_response_state: {
            version: 1,
            scope: 'test-scope',
            items: [],
          },
        },
      },
      {
        type: 'assistant',
        uuid: 'assistant-tool',
        timestamp: '2026-01-01T00:00:00.000Z',
        message: {
          ...assistantMessage,
          content: [
            {
              type: 'tool_use',
              id: 'search-1',
              name: 'ToolSearch',
              input: { query: 'repository' },
            },
          ],
          _openai_response_state: {
            version: 1,
            scope: 'test-scope',
            items: responseItems,
          },
        },
      },
    ] as never)

    expect(normalized).toHaveLength(1)
    expect(
      (
        normalized[0]!.message as unknown as Record<string, unknown>
      )._openai_response_state,
    ).toEqual({ version: 1, scope: 'test-scope', items: responseItems })
  })

  it('keeps opaque state on a reasoning-only Responses message', () => {
    const responseItems = [
      { type: 'reasoning', id: 'reasoning-1', encrypted_content: 'opaque' },
    ]
    const normalized = normalizeMessagesForAPI([
      {
        type: 'assistant',
        uuid: 'assistant-reasoning',
        timestamp: '2026-01-01T00:00:00.000Z',
        message: {
          id: 'response-1',
          type: 'message',
          role: 'assistant',
          model: 'gpt-test',
          stop_reason: 'end_turn',
          stop_sequence: null,
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
          },
          content: [
            { type: 'thinking', thinking: 'Summary', signature: '' },
          ],
          _openai_response_state: {
            version: 1,
            scope: 'test-scope',
            items: responseItems,
          },
        },
      },
    ] as never)

    expect(normalized).toHaveLength(1)
    expect(
      (normalized[0]!.message as unknown as Record<string, unknown>)
        ._openai_response_state,
    ).toEqual({ version: 1, scope: 'test-scope', items: responseItems })
  })

  it('migrates legacy unscoped Responses state without replaying it', () => {
    const normalized = normalizeMessagesForAPI([
      {
        type: 'assistant',
        uuid: 'assistant-legacy',
        timestamp: '2026-01-01T00:00:00.000Z',
        message: {
          id: 'response-legacy',
          type: 'message',
          role: 'assistant',
          model: 'gpt-test',
          stop_reason: 'end_turn',
          stop_sequence: null,
          usage: { input_tokens: 1, output_tokens: 1 },
          content: [
            { type: 'thinking', thinking: 'Legacy summary', signature: '' },
            { type: 'text', text: 'Visible answer' },
          ],
          _openai_response_items: [
            { type: 'reasoning', id: 'reasoning-1', encrypted_content: 'opaque' },
          ],
        },
      },
    ] as never)

    expect(normalized).toHaveLength(1)
    expect(normalized[0]!.message.content).toEqual([
      { type: 'text', text: 'Legacy summary\n\n' },
      { type: 'text', text: 'Visible answer' },
    ])
    const message = normalized[0]!.message as unknown as Record<string, unknown>
    expect(message._openai_response_items).toBeUndefined()
    expect(message._openai_response_state).toBeUndefined()
  })

  it('converts Responses reasoning summaries to text for Anthropic', () => {
    delete process.env.OPENAI_API_KEY
    process.env.ANTHROPIC_API_KEY = 'test-key'
    const normalized = normalizeMessagesForAPI([
      {
        type: 'assistant',
        uuid: 'assistant-mixed',
        timestamp: '2026-01-01T00:00:00.000Z',
        message: {
          id: 'response-1',
          type: 'message',
          role: 'assistant',
          model: 'gpt-test',
          stop_reason: 'end_turn',
          stop_sequence: null,
          usage: { input_tokens: 1, output_tokens: 1 },
          content: [
            { type: 'thinking', thinking: 'Reasoning summary', signature: '' },
            { type: 'text', text: 'Visible answer' },
          ],
          _openai_response_state: {
            version: 1,
            scope: 'test-scope',
            items: [
              { type: 'reasoning', id: 'reasoning-1', encrypted_content: 'opaque' },
            ],
          },
        },
      },
    ] as never)

    expect(normalized).toHaveLength(1)
    expect(normalized[0]!.message.content).toEqual([
      { type: 'text', text: 'Reasoning summary' },
      { type: 'text', text: 'Visible answer' },
    ])
    expect(
      (normalized[0]!.message as unknown as Record<string, unknown>)
        ._openai_response_state,
    ).toBeUndefined()
  })
})
