import { describe, expect, test } from 'bun:test'
import {
  GPT_5_6_AUTO_COMPACT_THRESHOLD,
  GPT_5_6_EFFECTIVE_CONTEXT_WINDOW,
} from '../../utils/context.js'
import {
  calculateTokenWarningState,
  getAutoCompactThreshold,
  getEffectiveContextWindowSize,
} from './autoCompact.js'

describe('OpenAI-compatible model compaction budgets', () => {
  test.each([
    'gpt-5.6-sol',
    'gpt-5.6-terra(high)',
    'gpt-5.6-luna(xhigh)',
  ])('%s uses the Codex-aligned compaction budget', model => {
    expect(getEffectiveContextWindowSize(model)).toBe(
      GPT_5_6_EFFECTIVE_CONTEXT_WINDOW,
    )
    expect(getAutoCompactThreshold(model)).toBe(
      GPT_5_6_AUTO_COMPACT_THRESHOLD,
    )

    expect(
      calculateTokenWarningState(GPT_5_6_AUTO_COMPACT_THRESHOLD - 1, model),
    ).toMatchObject({ isAboveAutoCompactThreshold: false })
    expect(
      calculateTokenWarningState(GPT_5_6_AUTO_COMPACT_THRESHOLD, model),
    ).toMatchObject({ isAboveAutoCompactThreshold: true })
  })

  test('keeps the legacy compaction budget for unknown compatible models', () => {
    expect(getEffectiveContextWindowSize('openrouter/custom-model')).toBe(
      180_000,
    )
    expect(getAutoCompactThreshold('openrouter/custom-model')).toBe(167_000)
  })
})
