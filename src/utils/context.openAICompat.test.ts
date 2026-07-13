import { describe, expect, test } from 'bun:test'
import {
  GPT_5_6_CONTEXT_WINDOW,
  getContextWindowForModel,
  isGpt56Model,
  MODEL_CONTEXT_WINDOW_DEFAULT,
} from './context.js'

describe('OpenAI-compatible model context windows', () => {
  test.each([
    'gpt-5.6-sol',
    'gpt-5.6-terra',
    'gpt-5.6-luna',
    'gpt-5.6-sol(high)',
    'gpt-5.6-luna(xhigh)',
  ])('%s uses the GPT-5.6 client context budget', model => {
    expect(isGpt56Model(model)).toBe(true)
    expect(getContextWindowForModel(model)).toBe(GPT_5_6_CONTEXT_WINDOW)
  })

  test('does not apply the GPT-5.6 budget to unknown compatible models', () => {
    expect(isGpt56Model('openrouter/custom-model')).toBe(false)
    expect(getContextWindowForModel('openrouter/custom-model')).toBe(
      MODEL_CONTEXT_WINDOW_DEFAULT,
    )
  })
})
