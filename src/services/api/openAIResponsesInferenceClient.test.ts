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
      { type: 'function_call', call_id: 'c1', name: 'Read', arguments: '{}' },
    ]
    const request = buildOpenAIResponsesRequest({
      model: 'gpt-test',
      max_tokens: 10,
      messages: [
        {
          role: 'assistant',
          content: [],
          _openai_response_items: items,
        },
      ],
    } as never)
    expect(request.input).toEqual(items)
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
  })

  it('uses the standalone compact endpoint and returns its canonical output', async () => {
    let url = ''
    const client = new OpenAIResponsesInferenceClient({
      baseURL: 'https://proxy.example.test/v1',
      fetch: async input => {
        url = String(input)
        return Response.json({ object: 'response.compaction', output: [{ type: 'compaction', encrypted_content: 'opaque' }] })
      },
    })
    const compacted = await client.compactResponse({
      model: 'gpt-test',
      max_tokens: 100,
      messages: [{ role: 'user', content: 'history' }],
    })
    expect(url).toBe('https://proxy.example.test/v1/responses/compact')
    expect(compacted.output).toEqual([{ type: 'compaction', encrypted_content: 'opaque' }])
  })
})
