import { describe, expect, it } from 'bun:test'
import {
  buildOpenAIResponsesRequest,
  createOpenAIResponsesStateScope,
  OpenAIResponsesInferenceClient,
} from './openAIResponsesInferenceClient.js'
import {
  isOpenAIResponsesRetryableError,
  OpenAIResponsesResponseError,
} from './openAICompatInferenceClient.js'

describe('OpenAIResponsesInferenceClient', () => {
  it('maps Anthropic-shaped tools and messages to Responses items', () => {
    const request = buildOpenAIResponsesRequest({
      model: 'gpt-test',
      max_tokens: 100,
      system: [{ type: 'text', text: 'system' }],
      messages: [
        { role: 'user', content: 'hello' },
        {
          role: 'assistant',
          content: [
            { type: 'tool_use', id: 'call-1', name: 'Read', input: { file_path: 'a.ts' } },
          ],
        },
        {
          role: 'user',
          content: [{ type: 'tool_result', tool_use_id: 'call-1', content: 'file' }],
        },
      ],
      tools: [
        {
          name: 'Read',
          description: 'Read a file',
          input_schema: { type: 'object', properties: { file_path: { type: 'string' } }, required: ['file_path'] },
        },
      ],
    } as never)

    expect(request).toMatchObject({
      model: 'gpt-test',
      instructions: 'system',
      store: false,
      input: [
        { type: 'message', role: 'user', content: 'hello' },
        { type: 'function_call', call_id: 'call-1', name: 'Read' },
        { type: 'function_call_output', call_id: 'call-1', output: 'file' },
      ],
      tools: [{ type: 'function', name: 'Read' }],
    })
  })

  it('keeps the local search function immediate without exposing deferred schemas', () => {
    const request = buildOpenAIResponsesRequest({
      model: 'gpt-5.4',
      max_tokens: 100,
      messages: [{ role: 'user', content: 'find the repository tool' }],
      tools: [
        {
          name: 'Read',
          description: 'Read a file',
          input_schema: { type: 'object', properties: {} },
        },
        {
          name: 'RepositorySearch',
          description: 'Search a repository',
          input_schema: {
            type: 'object',
            properties: { query: { type: 'string' } },
            required: ['query'],
          },
          defer_loading: true,
        },
        {
          name: 'ToolSearch',
          description: 'Find deferred tools',
          input_schema: {
            type: 'object',
            properties: { query: { type: 'string' } },
            required: ['query'],
            additionalProperties: false,
          },
        },
      ],
    } as never)

    expect(request.tools).toEqual([
      {
        type: 'function',
        name: 'Read',
        description: 'Read a file',
        parameters: { type: 'object', properties: {} },
      },
      {
        type: 'function',
        name: 'ToolSearch',
        description: 'Find deferred tools',
        parameters: {
          type: 'object',
          properties: { query: { type: 'string' } },
          required: ['query'],
          additionalProperties: false,
        },
      },
    ])
  })

  it('rejects deferred tools without a client search executor', () => {
    expect(() =>
      buildOpenAIResponsesRequest({
        model: 'gpt-5.4',
        max_tokens: 100,
        messages: [{ role: 'user', content: 'find a tool' }],
        tools: [
          {
            name: 'RepositorySearch',
            description: 'Search a repository',
            input_schema: { type: 'object', properties: {} },
            defer_loading: true,
          },
        ],
      } as never),
    ).toThrow('Deferred Responses tools require the NCode ToolSearch tool')
  })

  it('records locally loaded tools as client tool-search items', async () => {
    const scope = createOpenAIResponsesStateScope(
      'https://proxy.example.test/v1/responses',
      'gpt-5.4',
    )
    const output = [
      {
        type: 'function_call',
        id: 'fc-search-call-1',
        call_id: 'search-call-1',
        name: 'ToolSearch',
        arguments: '{"query":"repository"}',
      },
    ]
    const client = new OpenAIResponsesInferenceClient({
      baseURL: 'https://proxy.example.test/v1',
      fetch: async () =>
        Response.json({
          id: 'response-1',
          model: 'gpt-5.4',
          status: 'completed',
          output,
        }),
    })

    const assistant = await client.createMessage({
      model: 'gpt-5.4',
      max_tokens: 100,
      messages: [{ role: 'user', content: 'search the repository' }],
    })
    expect(assistant.content).toEqual([
      {
        type: 'tool_use',
        id: 'search-call-1',
        name: 'ToolSearch',
        input: { query: 'repository' },
      },
    ])
    const nextRequest = buildOpenAIResponsesRequest({
      model: 'gpt-5.4',
      max_tokens: 100,
      messages: [
        assistant,
        {
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'search-call-1',
              content: [
                {
                  type: 'tool_reference',
                  tool_name: 'RepositorySearch',
                },
              ],
            },
          ],
        },
      ],
      tools: [
        {
          name: 'RepositorySearch',
          description: 'Search a repository',
          input_schema: { type: 'object', properties: {} },
          defer_loading: true,
        },
        {
          name: 'ToolSearch',
          description: 'Find deferred tools',
          input_schema: {
            type: 'object',
            properties: { query: { type: 'string' } },
            required: ['query'],
          },
        },
      ],
    } as never, scope)

    expect(nextRequest.input).toEqual([
      ...output,
      {
        type: 'function_call_output',
        call_id: 'search-call-1',
        output: 'Loaded tools: RepositorySearch',
      },
      {
        type: 'tool_search_call',
        call_id: 'ncode_tool_load_e19ccfda1cdd13a9f872fcd7',
        status: 'completed',
        execution: 'client',
        arguments: { query: 'RepositorySearch', limit: 1 },
      },
      {
        type: 'tool_search_output',
        call_id: 'ncode_tool_load_e19ccfda1cdd13a9f872fcd7',
        status: 'completed',
        execution: 'client',
        tools: [
          {
            type: 'function',
            name: 'RepositorySearch',
            description: 'Search a repository',
            parameters: { type: 'object', properties: {} },
            defer_loading: true,
          },
        ],
      },
    ])
    expect(nextRequest.tools).toEqual([
      {
        type: 'function',
        name: 'ToolSearch',
        description: 'Find deferred tools',
        parameters: {
          type: 'object',
          properties: { query: { type: 'string' } },
          required: ['query'],
        },
      },
    ])
  })

  it('replays native output items without translating them', () => {
    const scope = createOpenAIResponsesStateScope(
      'https://proxy.example.test/v1/responses',
      'gpt-5.4',
      { Authorization: 'Bearer test' },
    )
    const items = [
      { type: 'reasoning', id: 'r1', encrypted_content: 'opaque' },
      {
        type: 'tool_search_call',
        id: 'tsc1',
        call_id: 'tsc1',
        execution: 'client',
        status: 'completed',
        arguments: { query: 'read' },
      },
      {
        type: 'tool_search_output',
        id: 'tso1',
        call_id: 'tsc1',
        execution: 'client',
        status: 'completed',
        tools: [{ type: 'function', name: 'Read' }],
      },
      {
        type: 'function_call',
        call_id: 'c1',
        name: 'Read',
        arguments: '{}',
      },
      {
        type: 'function_call',
        call_id: 'c2',
        name: 'Glob',
        arguments: '{}',
      },
      {
        type: 'message',
        id: 'm1',
        role: 'assistant',
        content: [{ type: 'output_text', text: 'done' }],
      },
    ]
    const request = buildOpenAIResponsesRequest({
      model: 'gpt-5.4',
      max_tokens: 10,
      messages: Array.from({ length: 4 }, () => ({
        role: 'assistant',
        content: [],
        _openai_response_state: { version: 1, scope, items },
      })),
    } as never, scope)
    expect(request.input).toEqual(items)
  })

  it('does not replay tool-search items to an unsupported Responses model', () => {
    const scope = createOpenAIResponsesStateScope(
      'https://proxy.example.test/v1/responses',
      'gpt-5.3',
    )
    const request = buildOpenAIResponsesRequest({
      model: 'gpt-5.3',
      max_tokens: 10,
      messages: [
        {
          role: 'assistant',
          content: [],
          _openai_response_state: {
            version: 1,
            scope,
            items: [
              { type: 'reasoning', id: 'r1', encrypted_content: 'opaque' },
              { type: 'tool_search_call', call_id: 'ts1', arguments: {} },
              { type: 'tool_search_output', call_id: 'ts1', tools: [] },
              {
                type: 'message',
                role: 'assistant',
                content: [{ type: 'output_text', text: 'done' }],
              },
            ],
          },
        },
      ],
    } as never, scope)

    expect(request.input).toEqual([
      { type: 'reasoning', id: 'r1', encrypted_content: 'opaque' },
      {
        type: 'message',
        role: 'assistant',
        content: [{ type: 'output_text', text: 'done' }],
      },
    ])
  })

  it('replays opaque state only for the endpoint, model, and credentials that created it', () => {
    const endpoint = 'https://proxy.example.test/v1/responses'
    const originalScope = createOpenAIResponsesStateScope(
      endpoint,
      'gpt-test',
      { Authorization: 'Bearer original-secret' },
    )
    const changedCredentialScope = createOpenAIResponsesStateScope(
      endpoint,
      'gpt-test',
      { Authorization: 'Bearer replacement-secret' },
    )
    const changedModelScope = createOpenAIResponsesStateScope(
      endpoint,
      'gpt-other',
      { Authorization: 'Bearer original-secret' },
    )
    const changedEndpointScope = createOpenAIResponsesStateScope(
      'https://other.example.test/v1/responses',
      'gpt-test',
      { Authorization: 'Bearer original-secret' },
    )

    expect(new Set([
      originalScope,
      changedCredentialScope,
      changedModelScope,
      changedEndpointScope,
    ]).size).toBe(4)
    expect(originalScope).not.toContain('original-secret')

    const request = buildOpenAIResponsesRequest({
      model: 'gpt-test',
      max_tokens: 10,
      messages: [
        {
          role: 'assistant',
          content: [
            { type: 'thinking', thinking: 'Reasoning summary. ', signature: '' },
            { type: 'text', text: 'Visible answer.' },
            { type: 'tool_use', id: 'call-1', name: 'Read', input: { path: 'a.ts' } },
          ],
          _openai_response_state: {
            version: 1,
            scope: originalScope,
            items: [{ type: 'reasoning', id: 'r1', encrypted_content: 'opaque' }],
          },
        },
      ],
    } as never, changedCredentialScope)

    expect(request.input).toEqual([
      {
        type: 'message',
        role: 'assistant',
        content: 'Reasoning summary. Visible answer.',
      },
      {
        type: 'function_call',
        call_id: 'call-1',
        name: 'Read',
        arguments: '{"path":"a.ts"}',
      },
    ])
  })

  it('never replays legacy unscoped state and preserves its readable content', () => {
    const request = buildOpenAIResponsesRequest({
      model: 'gpt-test',
      max_tokens: 10,
      messages: [
        {
          role: 'assistant',
          content: [
            { type: 'thinking', thinking: 'Legacy summary. ', signature: '' },
            { type: 'text', text: 'Visible answer.' },
          ],
          _openai_response_items: [
            { type: 'reasoning', id: 'r1', encrypted_content: 'opaque' },
          ],
        },
      ],
    } as never)

    expect(request.input).toEqual([
      {
        type: 'message',
        role: 'assistant',
        content: 'Legacy summary. Visible answer.',
      },
    ])
  })

  it('only sends reasoning configuration when thinking is enabled', () => {
    const base = {
      model: 'gpt-test',
      max_tokens: 10,
      messages: [{ role: 'user', content: 'hello' }],
    }
    const omitted = buildOpenAIResponsesRequest(base as never)
    expect(omitted).not.toHaveProperty('reasoning')
    expect(omitted).not.toHaveProperty('include')

    const disabled = buildOpenAIResponsesRequest({
      ...base,
      thinking: { type: 'disabled' },
      output_config: { effort: 'xhigh' },
    } as never)
    expect(disabled.reasoning).toEqual({ effort: 'none' })
    expect(disabled).not.toHaveProperty('include')

    const enabled = buildOpenAIResponsesRequest({
      ...base,
      thinking: { type: 'adaptive' },
    } as never)
    expect(enabled.reasoning).toEqual({ effort: 'high', summary: 'auto' })
    expect(enabled.include).toEqual(['reasoning.encrypted_content'])

    const explicit = buildOpenAIResponsesRequest({
      ...base,
      thinking: { type: 'adaptive' },
      output_config: { effort: 'xhigh' },
    } as never)
    expect(explicit.reasoning).toEqual({ effort: 'xhigh', summary: 'auto' })
  })

  it('preserves structured output and tool-use constraints', () => {
    const schemaRequest = buildOpenAIResponsesRequest({
      model: 'gpt-test',
      max_tokens: 10,
      messages: [{ role: 'user', content: 'json' }],
      output_config: {
        format: {
          type: 'json_schema',
          name: 'Result',
          schema: { type: 'object', properties: { ok: { type: 'boolean' } } },
          strict: true,
        },
      },
      tool_choice: { type: 'auto', disable_parallel_tool_use: true },
    } as never)
    expect(schemaRequest).toMatchObject({
      parallel_tool_calls: false,
      text: {
        format: {
          type: 'json_schema',
          name: 'Result',
          strict: true,
        },
      },
    })

    const objectRequest = buildOpenAIResponsesRequest({
      model: 'gpt-test',
      max_tokens: 10,
      messages: [{ role: 'user', content: 'json' }],
      output_config: { format: { type: 'json_object' } },
      tool_choice: { type: 'none' },
    } as never)
    expect(objectRequest).toMatchObject({
      tool_choice: 'none',
      text: { format: { type: 'json_object' } },
    })
  })

  it('translates image and document inputs and rejects unsupported blocks', () => {
    const request = buildOpenAIResponsesRequest({
      model: 'gpt-test',
      max_tokens: 10,
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: 'inspect these' },
            {
              type: 'image',
              source: { type: 'base64', media_type: 'image/png', data: 'aW1n' },
            },
            {
              type: 'document',
              source: { type: 'base64', media_type: 'application/pdf', data: 'cGRm' },
            },
          ],
        },
      ],
    } as never)
    expect(request.input).toEqual([
      {
        type: 'message',
        role: 'user',
        content: [
          { type: 'input_text', text: 'inspect these' },
          { type: 'input_image', image_url: 'data:image/png;base64,aW1n' },
          {
            type: 'input_file',
            filename: 'document.pdf',
            file_data: 'data:application/pdf;base64,cGRm',
          },
        ],
      },
    ])

    expect(() =>
      buildOpenAIResponsesRequest({
        model: 'gpt-test',
        max_tokens: 10,
        messages: [{ role: 'user', content: [{ type: 'audio', data: 'x' }] }],
      } as never),
    ).toThrow('Unsupported Responses input content block: audio')
  })

  it('maps unary output and normalizes cached usage', async () => {
    let body: Record<string, unknown> | undefined
    const client = new OpenAIResponsesInferenceClient({
      baseURL: 'https://proxy.example.test/v1',
      headers: { Authorization: 'Bearer test' },
      fetch: async (_url, init) => {
        body = JSON.parse(String(init?.body))
        return new Response(
          JSON.stringify({
            id: 'resp-1',
            model: 'gpt-test',
            status: 'completed',
            output: [
              { type: 'function_call', call_id: 'call-1', name: 'Read', arguments: '{"file_path":"a.ts"}' },
            ],
            usage: { input_tokens: 100, input_tokens_details: { cached_tokens: 60 }, output_tokens: 10 },
          }),
          { status: 200, headers: { 'x-request-id': 'req-1' } },
        )
      },
    })

    const result = await client.createMessage({
      model: 'gpt-test',
      max_tokens: 100,
      messages: [{ role: 'user', content: 'read' }],
      thinking: { type: 'enabled', budget_tokens: 1024 },
    })
    expect(body).toMatchObject({ store: false, include: ['reasoning.encrypted_content'] })
    expect(result.content[0]).toMatchObject({ type: 'tool_use', id: 'call-1', name: 'Read' })
    expect(result.usage).toMatchObject({ input_tokens: 40, cache_read_input_tokens: 60, output_tokens: 10 })
    expect(
      (
        (result as unknown as Record<string, unknown>)
          ._openai_response_state as { items: unknown[] }
      ).items,
    ).toHaveLength(1)
  })

  it('classifies terminal unary response errors without treating them as transport failures', async () => {
    const responses = [
      {
        id: 'resp-failed',
        model: 'gpt-test',
        status: 'failed',
        error: {
          code: 'server_error',
          type: 'server_error',
          message: 'Upstream inference failed',
        },
        output: [],
      },
      {
        id: 'resp-invalid',
        model: 'gpt-test',
        status: 'failed',
        error: {
          code: 'invalid_prompt',
          type: 'invalid_request_error',
          message: 'Prompt is invalid',
        },
        output: [],
      },
      {
        id: 'resp-cancelled',
        model: 'gpt-test',
        status: 'cancelled',
        error: { code: 'cancelled', message: 'Request was cancelled' },
        output: [],
      },
    ]
    const client = new OpenAIResponsesInferenceClient({
      baseURL: 'https://proxy.example.test/v1',
      fetch: async () => Response.json(responses.shift()),
    })
    const params = {
      model: 'gpt-test',
      max_tokens: 10,
      messages: [{ role: 'user', content: 'request' }],
    } as const

    const serverError = await client.createMessage(params).catch(error => error)
    expect(serverError).toBeInstanceOf(OpenAIResponsesResponseError)
    expect(isOpenAIResponsesRetryableError(serverError)).toBe(true)
    expect(
      (serverError as OpenAIResponsesResponseError).telemetryMessage,
    ).toBe('openai_responses_error code=server_error type=server_error')
    expect(
      (serverError as OpenAIResponsesResponseError).telemetryMessage,
    ).not.toContain('Upstream inference failed')

    const invalidPrompt = await client.createMessage(params).catch(error => error)
    expect(invalidPrompt).toBeInstanceOf(OpenAIResponsesResponseError)
    expect(isOpenAIResponsesRetryableError(invalidPrompt)).toBe(false)

    const cancelled = await client.createMessage(params).catch(error => error)
    expect(cancelled).toBeInstanceOf(OpenAIResponsesResponseError)
    expect(isOpenAIResponsesRetryableError(cancelled)).toBe(false)
  })

  it('maps unary refusals to the downstream refusal contract', async () => {
    const client = new OpenAIResponsesInferenceClient({
      baseURL: 'https://proxy.example.test/v1',
      fetch: async () =>
        Response.json({
          id: 'resp-refusal',
          model: 'gpt-test',
          status: 'completed',
          output: [
            {
              type: 'message',
              content: [{ type: 'refusal', refusal: 'I cannot help with that.' }],
            },
          ],
        }),
    })
    const result = await client.createMessage({
      model: 'gpt-test',
      max_tokens: 10,
      messages: [{ role: 'user', content: 'request' }],
    })
    expect(result.stop_reason).toBe('refusal')
    expect(result.content).toEqual([
      { type: 'text', text: 'I cannot help with that.' },
    ])
  })

  it('distinguishes content filtering from output-token exhaustion', async () => {
    const responses = [
      {
        id: 'resp-filtered',
        model: 'gpt-test',
        status: 'incomplete',
        incomplete_details: { reason: 'content_filter' },
        output: [],
      },
      {
        id: 'resp-limited',
        model: 'gpt-test',
        status: 'incomplete',
        incomplete_details: { reason: 'max_output_tokens' },
        output: [],
      },
    ]
    const client = new OpenAIResponsesInferenceClient({
      baseURL: 'https://proxy.example.test/v1',
      fetch: async () => Response.json(responses.shift()),
    })
    const params = {
      model: 'gpt-test',
      max_tokens: 10,
      messages: [{ role: 'user', content: 'request' }],
    } as const

    expect((await client.createMessage(params)).stop_reason).toBe('refusal')
    expect((await client.createMessage(params)).stop_reason).toBe('max_tokens')
  })

  it('forwards an already-aborted request signal', async () => {
    const abortController = new AbortController()
    abortController.abort(new Error('cancelled'))
    let requestWasAborted = false
    const client = new OpenAIResponsesInferenceClient({
      baseURL: 'https://proxy.example.test/v1',
      fetch: async (_input, init) => {
        requestWasAborted = init?.signal?.aborted === true
        throw init?.signal?.reason
      },
    })

    let thrown: unknown
    try {
      await client.createMessage(
        {
          model: 'gpt-test',
          max_tokens: 10,
          messages: [{ role: 'user', content: 'request' }],
        },
        { signal: abortController.signal },
      )
    } catch (error) {
      thrown = error
    }
    expect(requestWasAborted).toBe(true)
    expect((thrown as Error).message).toBe('cancelled')
  })

  it('streams typed text and function-call events while retaining raw items', async () => {
    const output = [
      { type: 'function_call', id: 'fc-1', call_id: 'call-1', name: 'Read', arguments: '{"file_path":"a.ts"}', status: 'completed' },
    ]
    const frames = [
      { type: 'response.created', response: { id: 'resp-1', model: 'gpt-test' } },
      { type: 'response.output_item.added', output_index: 0, item: { type: 'function_call', id: 'fc-1', call_id: 'call-1', name: 'Read', arguments: '' } },
      { type: 'response.function_call_arguments.delta', output_index: 0, delta: '{"file_path":"a.ts"}' },
      { type: 'response.output_item.done', output_index: 0, item: output[0] },
      { type: 'response.completed', response: { id: 'resp-1', model: 'gpt-test', status: 'completed', output, usage: { input_tokens: 12, output_tokens: 3 } } },
    ]
      .map(event => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`)
      .join('')
    const client = new OpenAIResponsesInferenceClient({
      baseURL: 'https://proxy.example.test/v1',
      fetch: async () => new Response(frames, { headers: { 'content-type': 'text/event-stream' } }),
    })
    const stream = await client.createMessage({
      model: 'gpt-test',
      max_tokens: 100,
      messages: [{ role: 'user', content: 'read' }],
      stream: true,
    })
    const events = []
    for await (const event of stream) events.push(event)
    expect(events).toContainEqual(expect.objectContaining({
      type: 'content_block_start',
      content_block: expect.objectContaining({ type: 'tool_use', id: 'call-1', name: 'Read' }),
    }))
    const start = events[0] as unknown as { message: Record<string, unknown> }
    expect(
      (start.message._openai_response_state as { items: unknown[] }).items,
    ).toEqual(output)
    expect(start.message.usage).toMatchObject({
      input_tokens: 12,
      output_tokens: 3,
    })
  })

  it('surfaces structured API error details', async () => {
    const client = new OpenAIResponsesInferenceClient({
      baseURL: 'https://proxy.example.test/v1',
      fetch: async () =>
        Response.json(
          { error: { message: 'Unsupported tool_search execution mode' } },
          { status: 400, statusText: 'Bad Request' },
        ),
    })

    expect(
      client.createMessage({
        model: 'gpt-test',
        max_tokens: 10,
        messages: [{ role: 'user', content: 'request' }],
      }),
    ).rejects.toThrow(
      '400 Bad Request: Unsupported tool_search execution mode',
    )
  })

  it('rejects a stream that reaches EOF without a terminal event', async () => {
    const frames = [
      { type: 'response.created', response: { id: 'resp-1', model: 'gpt-test' } },
      { type: 'response.output_text.delta', delta: 'partial' },
    ]
      .map(event => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`)
      .join('')
    const client = new OpenAIResponsesInferenceClient({
      baseURL: 'https://proxy.example.test/v1',
      fetch: async () => new Response(frames),
    })
    const stream = await client.createMessage({
      model: 'gpt-test',
      max_tokens: 10,
      messages: [{ role: 'user', content: 'request' }],
      stream: true,
    })

    let thrown: unknown
    try {
      for await (const _event of stream) {
        // Consume the stream to its terminal validation.
      }
    } catch (error) {
      thrown = error
    }
    expect(thrown).toBeInstanceOf(Error)
    expect((thrown as Error).message).toContain(
      'ended before a terminal response event',
    )
  })

  it('surfaces terminal stream failures as semantic Responses errors', async () => {
    const failed = {
      type: 'response.failed',
      response: {
        id: 'response-failed',
        model: 'gpt-test',
        status: 'failed',
        error: {
          code: 'invalid_prompt',
          type: 'invalid_request_error',
          message: 'Prompt is invalid',
        },
        output: [],
      },
    }
    const client = new OpenAIResponsesInferenceClient({
      baseURL: 'https://proxy.example.test/v1',
      fetch: async () =>
        new Response(
          `event: response.failed\ndata: ${JSON.stringify(failed)}\n\n`,
        ),
    })
    const stream = await client.createMessage({
      model: 'gpt-test',
      max_tokens: 10,
      messages: [{ role: 'user', content: 'request' }],
      stream: true,
    })

    let thrown: unknown
    try {
      for await (const _event of stream) {
        // Consume the stream to its terminal validation.
      }
    } catch (error) {
      thrown = error
    }

    expect(thrown).toBeInstanceOf(OpenAIResponsesResponseError)
    expect(isOpenAIResponsesRetryableError(thrown)).toBe(false)
  })

  it('does not release a truncated function call for execution', async () => {
    const frames = [
      { type: 'response.created', response: { id: 'resp-1', model: 'gpt-test' } },
      {
        type: 'response.output_item.added',
        output_index: 0,
        item: {
          type: 'function_call',
          id: 'fc-1',
          call_id: 'call-1',
          name: 'Write',
          arguments: '',
        },
      },
      {
        type: 'response.function_call_arguments.delta',
        output_index: 0,
        delta: '{"file_path":"a.ts"}',
      },
      {
        type: 'response.output_item.done',
        output_index: 0,
        item: {
          type: 'function_call',
          id: 'fc-1',
          call_id: 'call-1',
          name: 'Write',
          arguments: '{"file_path":"a.ts"}',
        },
      },
    ]
      .map(event => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`)
      .join('')
    const client = new OpenAIResponsesInferenceClient({
      baseURL: 'https://proxy.example.test/v1',
      fetch: async () => new Response(frames),
    })
    const stream = await client.createMessage({
      model: 'gpt-test',
      max_tokens: 10,
      messages: [{ role: 'user', content: 'request' }],
      stream: true,
    })

    const events = []
    let thrown: unknown
    try {
      for await (const event of stream) events.push(event)
    } catch (error) {
      thrown = error
    }
    expect(thrown).toBeInstanceOf(Error)
    expect(events).not.toContainEqual(
      expect.objectContaining({ type: 'content_block_stop' }),
    )
  })

  it('retains opaque items for a response with no visible content', async () => {
    const output = [
      { type: 'reasoning', id: 'r1', encrypted_content: 'opaque', summary: [] },
    ]
    const frames = [
      { type: 'response.created', response: { id: 'resp-1', model: 'gpt-test' } },
      {
        type: 'response.completed',
        response: {
          id: 'resp-1',
          model: 'gpt-test',
          status: 'completed',
          output,
          usage: { input_tokens: 5, output_tokens: 2 },
        },
      },
    ]
      .map(event => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`)
      .join('')
    const client = new OpenAIResponsesInferenceClient({
      baseURL: 'https://proxy.example.test/v1',
      fetch: async () => new Response(frames),
    })
    const stream = await client.createMessage({
      model: 'gpt-test',
      max_tokens: 10,
      messages: [{ role: 'user', content: 'request' }],
      stream: true,
    })

    const events = []
    for await (const event of stream) events.push(event)
    const start = events[0] as unknown as { message: Record<string, unknown> }
    expect(
      (start.message._openai_response_state as { items: unknown[] }).items,
    ).toEqual(output)
    expect(events).toContainEqual(
      expect.objectContaining({
        type: 'message_delta',
        delta: expect.objectContaining({ stop_reason: 'end_turn' }),
      }),
    )
  })

  it('streams refusal text with a refusal stop reason', async () => {
    const output = [
      {
        type: 'message',
        content: [{ type: 'refusal', refusal: 'Cannot comply.' }],
      },
    ]
    const frames = [
      { type: 'response.created', response: { id: 'resp-1', model: 'gpt-test' } },
      { type: 'response.refusal.delta', delta: 'Cannot comply.' },
      {
        type: 'response.completed',
        response: {
          id: 'resp-1',
          model: 'gpt-test',
          status: 'completed',
          output,
          usage: { input_tokens: 5, output_tokens: 2 },
        },
      },
    ]
      .map(event => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`)
      .join('')
    const client = new OpenAIResponsesInferenceClient({
      baseURL: 'https://proxy.example.test/v1',
      fetch: async () => new Response(frames),
    })
    const stream = await client.createMessage({
      model: 'gpt-test',
      max_tokens: 10,
      messages: [{ role: 'user', content: 'request' }],
      stream: true,
    })
    const events = []
    for await (const event of stream) events.push(event)
    expect(events).toContainEqual(
      expect.objectContaining({
        type: 'message_delta',
        delta: expect.objectContaining({ stop_reason: 'refusal' }),
      }),
    )
  })

  it('maps a streamed content filter to refusal instead of max tokens', async () => {
    const frames = [
      { type: 'response.created', response: { id: 'resp-1', model: 'gpt-test' } },
      {
        type: 'response.incomplete',
        response: {
          id: 'resp-1',
          model: 'gpt-test',
          status: 'incomplete',
          incomplete_details: { reason: 'content_filter' },
          output: [],
          usage: { input_tokens: 5, output_tokens: 0 },
        },
      },
    ]
      .map(event => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`)
      .join('')
    const client = new OpenAIResponsesInferenceClient({
      baseURL: 'https://proxy.example.test/v1',
      fetch: async () => new Response(frames),
    })
    const stream = await client.createMessage({
      model: 'gpt-test',
      max_tokens: 10,
      messages: [{ role: 'user', content: 'request' }],
      stream: true,
    })

    const events = []
    for await (const event of stream) events.push(event)
    expect(events).toContainEqual(
      expect.objectContaining({
        type: 'message_delta',
        delta: expect.objectContaining({ stop_reason: 'refusal' }),
      }),
    )
    expect(events).not.toContainEqual(
      expect.objectContaining({
        type: 'message_delta',
        delta: expect.objectContaining({ stop_reason: 'max_tokens' }),
      }),
    )
  })

  it('counts the complete request with the native input-token endpoint', async () => {
    let url = ''
    let body: Record<string, unknown> = {}
    const client = new OpenAIResponsesInferenceClient({
      baseURL: 'https://proxy.example.test/v1',
      fetch: async (input, init) => {
        url = String(input)
        body = JSON.parse(String(init?.body))
        return Response.json({
          object: 'response.input_tokens',
          input_tokens: 321,
        })
      },
    })
    const result = await client.countTokens({
      model: 'gpt-test',
      system: [{ type: 'text', text: 'system' }],
      messages: [{ role: 'user', content: 'request' }],
      tools: [
        {
          name: 'LargeTool',
          description: 'A large tool schema',
          input_schema: {
            type: 'object',
            properties: { payload: { type: 'string' } },
          },
        },
      ],
      thinking: { type: 'enabled', budget_tokens: 1024 },
    })
    expect(url).toBe('https://proxy.example.test/v1/responses/input_tokens')
    expect(body).toMatchObject({
      model: 'gpt-test',
      instructions: 'system',
      tools: [{ type: 'function', name: 'LargeTool' }],
    })
    expect(body).not.toHaveProperty('include')
    expect(body).not.toHaveProperty('max_output_tokens')
    expect(body).not.toHaveProperty('store')
    expect(result.input_tokens).toBe(321)
  })
})
