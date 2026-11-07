/**
 * 转写状态机的单测。纯函数，不需要浏览器也不需要装依赖：
 *   node --test tests/*.test.js
 */

import assert from 'node:assert/strict'
import { test } from 'node:test'

import {
  applyMessage,
  closeOpenSentence,
  createTranscriptState
} from '../src/services/transcript.js'

const LLM = { useLLM: true }
const NO_LLM = { useLLM: false }

test('partial 会新建预览条目，重复到达则覆盖', () => {
  const state = createTranscriptState()

  const first = applyMessage(state, { type: 'partial', segment_id: 's1', text: '你' }, NO_LLM)
  assert.equal(first.scroll, true)
  assert.equal(state.entries.length, 1)
  assert.equal(state.entries[0].corrected_text, '你')
  assert.equal(state.entries[0].status, 'partial')

  applyMessage(state, { type: 'partial', segment_id: 's1', text: '你好' }, NO_LLM)
  assert.equal(state.entries.length, 1)
  assert.equal(state.entries[0].corrected_text, '你好')
})

test('LLM 模式下有未收尾句子时忽略 partial', () => {
  const state = createTranscriptState()
  applyMessage(state, { type: 'transcription_raw', segment_id: 's1', raw_text: '第一句' }, LLM)
  assert.equal(state.openId, 's1')

  const result = applyMessage(state, { type: 'partial', segment_id: 's2', text: '新的' }, LLM)
  assert.equal(result.scroll, true)
  assert.equal(state.entries.length, 1, '不该多出一个预览框')
})

test('不开 LLM 时原始文本直接定稿', () => {
  const state = createTranscriptState()
  applyMessage(state, { type: 'transcription_raw', segment_id: 's1', raw_text: '一句话' }, NO_LLM)

  assert.equal(state.entries.length, 1)
  assert.equal(state.entries[0].status, 'final')
  assert.equal(state.openId, null)
})

test('开 LLM 时原始文本先标记为等纠错', () => {
  const state = createTranscriptState()
  applyMessage(state, { type: 'transcription_raw', segment_id: 's1', raw_text: '一句话' }, LLM)

  assert.equal(state.entries[0].status, 'whisper_done')
  assert.equal(state.openId, 's1')
  assert.equal(state.latestSegId, 's1')
})

test('新片段并入未收尾的句子，而不是新起一行', () => {
  const state = createTranscriptState()
  applyMessage(state, { type: 'transcription_raw', segment_id: 's1', raw_text: '今天讲' }, LLM)
  applyMessage(state, { type: 'llm_token', segment_id: 's1', token: '语音' })
  applyMessage(state, { type: 'llm_done', segment_id: 's1', corrected_text: '今天讲语音' }, LLM)
  assert.equal(state.entries[0].status, 'sentence_open')

  applyMessage(state, { type: 'transcription_raw', segment_id: 's2', raw_text: '识别' }, LLM)

  assert.equal(state.entries.length, 1, '应该还是同一行')
  assert.equal(state.entries[0].corrected_base, '今天讲语音')
  assert.equal(state.entries[0].raw_text, '今天讲识别')
  assert.equal(state.entries[0].corrected_text, '今天讲语音识别')
  assert.equal(state.latestSegId, 's2')
})

test('llm_token 只在片段匹配时累积', () => {
  const state = createTranscriptState()
  applyMessage(state, { type: 'transcription_raw', segment_id: 's1', raw_text: '原文' }, LLM)

  applyMessage(state, { type: 'llm_token', segment_id: 's1', token: '甲' })
  applyMessage(state, { type: 'llm_token', segment_id: 'old', token: '乙' })

  assert.equal(state.llmBuffer, '甲')
})

test('llm_done 拼接后以句号收尾则定稿', () => {
  const state = createTranscriptState()
  applyMessage(state, { type: 'transcription_raw', segment_id: 's1', raw_text: '原文' }, LLM)
  applyMessage(state, { type: 'llm_done', segment_id: 's1', corrected_text: '今天讲语音识别。' }, LLM)

  assert.equal(state.entries[0].corrected_text, '今天讲语音识别。')
  assert.equal(state.entries[0].status, 'final')
  assert.equal(state.openId, null)
  assert.equal(state.latestSegId, null)
  assert.equal(state.llmBuffer, '')
})

test('llm_done 没有句号则标记为未收尾', () => {
  const state = createTranscriptState()
  applyMessage(state, { type: 'transcription_raw', segment_id: 's1', raw_text: '原文' }, LLM)
  applyMessage(state, { type: 'llm_done', segment_id: 's1', corrected_text: '还没说完' }, LLM)

  assert.equal(state.entries[0].status, 'sentence_open')
  assert.equal(state.openId, 's1')
})

test('过期的 llm_done 被丢弃', () => {
  const state = createTranscriptState()
  applyMessage(state, { type: 'transcription_raw', segment_id: 's1', raw_text: '原文' }, LLM)
  applyMessage(state, { type: 'transcription_raw', segment_id: 's2', raw_text: '新片段' }, LLM)

  applyMessage(state, { type: 'llm_done', segment_id: 's1', corrected_text: '旧结果。' }, LLM)

  assert.equal(state.entries[0].status, 'whisper_done', '旧片段不该把句子定稿')
})

test('llm_done 没给文本时退回用缓冲的 token', () => {
  const state = createTranscriptState()
  applyMessage(state, { type: 'transcription_raw', segment_id: 's1', raw_text: '原文' }, LLM)
  applyMessage(state, { type: 'llm_token', segment_id: 's1', token: '纠错结果。' })
  applyMessage(state, { type: 'llm_done', segment_id: 's1' }, LLM)

  assert.equal(state.entries[0].corrected_text, '纠错结果。')
})

test('不开 LLM 时清理别的片段的残留预览', () => {
  const state = createTranscriptState()
  applyMessage(state, { type: 'partial', segment_id: 'p1', text: '旧预览' }, NO_LLM)
  applyMessage(state, { type: 'transcription_raw', segment_id: 's1', raw_text: '正式文本' }, NO_LLM)

  assert.equal(state.entries.length, 1)
  assert.equal(state.entries[0].segment_id, 's1')
})

test('error 消息返回文案给界面提示', () => {
  const state = createTranscriptState()
  const result = applyMessage(state, { type: 'error', message: '识别服务异常' }, LLM)
  assert.equal(result.error, '识别服务异常')
})

test('closeOpenSentence 把未收尾的句子定稿', () => {
  const state = createTranscriptState()
  applyMessage(state, { type: 'transcription_raw', segment_id: 's1', raw_text: '原文' }, LLM)
  assert.equal(closeOpenSentence(state), true)

  assert.equal(state.entries[0].status, 'final')
  assert.equal(state.openId, null)
  assert.equal(closeOpenSentence(state), false, '没有开着的句子就返回 false')
})
