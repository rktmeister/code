import { describe, expect, it } from 'bun:test'
import {
  buildOpenAIResponsesRequest,
  OpenAIResponsesInferenceClient,
} from './openAIResponsesInferenceClient.js'

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

  it('replays native output items without translating them', () => {
    const items = [
      { type: 'reasoning', id: 'r1', encrypted_content: 'opaque' },
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
      model: 'gpt-test',
      max_tokens: 10,
      messages: Array.from({ length: 4 }, () => ({
        role: 'assistant',
        content: [],
        _openai_response_items: items,
      })),
    } as never)
    expect(request.input).toEqual(items)
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
    })
    expect(body).toMatchObject({ store: false, include: ['reasoning.encrypted_content'] })
    expect(result.content[0]).toMatchObject({ type: 'tool_use', id: 'call-1', name: 'Read' })
    expect(result.usage).toMatchObject({ input_tokens: 40, cache_read_input_tokens: 60, output_tokens: 10 })
    expect((result as unknown as Record<string, unknown>)._openai_response_items).toHaveLength(1)
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
    expect(start.message._openai_response_items).toEqual(output)
    expect(start.message.usage).toMatchObject({
      input_tokens: 12,
      output_tokens: 3,
    })
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

  it('uses the standalone compact endpoint and returns its canonical output', async () => {
    let url = ''
    const client = new OpenAIResponsesInferenceClient({
      baseURL: 'https://proxy.example.test/v1',
      fetch: async input => {
        url = String(input)
        return Response.json({
          object: 'response.compaction',
          output: [{ type: 'compaction', encrypted_content: 'opaque' }],
          usage: { input_tokens: 100, output_tokens: 20 },
        })
      },
    })
    const compacted = await client.compactResponse({
      model: 'gpt-test',
      max_tokens: 100,
      messages: [{ role: 'user', content: 'history' }],
    })
    expect(url).toBe('https://proxy.example.test/v1/responses/compact')
    expect(compacted.output).toEqual([
      { type: 'compaction', encrypted_content: 'opaque' },
    ])
    expect(compacted.usage).toMatchObject({
      input_tokens: 100,
      output_tokens: 20,
    })
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
