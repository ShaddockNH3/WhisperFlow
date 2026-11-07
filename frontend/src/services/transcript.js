/**
 * 转写结果的状态机。
 *
 * 原来这是 App.vue 的 ws.onmessage 里一百多行 if/else，直接改 reactive 数组，
 * 想验证「LLM 模式下新片段应该追加到同一句而不是新起一行」只能靠人眼看。
 * 这里抽成纯函数：输入状态和一条消息，输出新状态，不碰 Vue、不碰 WebSocket。
 *
 * 一条字幕的生命周期：
 *   partial（临时预览） → whisper_done（原始文本已到，等纠错）
 *   → sentence_open（纠错到一半，句子还没结束） → final（句号收尾）
 */

export function createTranscriptState() {
  return {
    entries: [], // 按出现顺序排列的字幕条目
    llmBuffer: '', // 当前片段累积的 LLM token
    openId: null, // 还没收尾的句子所对应的 segment_id
    latestSegId: null // 最近一次喂给 LLM 的 segment_id，用来丢弃过期响应
  }
}

const SENTENCE_END = /[。！？…]$/

/**
 * 应用一条后端消息。
 *
 * @param {ReturnType<typeof createTranscriptState>} state 会被就地修改
 * @param {object} message 后端推送的 JSON
 * @param {{ useLLM: boolean }} options 当前是否开 AI 纠错
 * @returns {{ scroll: boolean, error?: string }}
 */
export function applyMessage(state, message, { useLLM = false } = {}) {
  switch (message.type) {
    case 'partial':
      return handlePartial(state, message, useLLM)
    case 'transcription_raw':
      return handleRaw(state, message, useLLM)
    case 'llm_token':
      return handleToken(state, message)
    case 'llm_done':
      return handleLlmDone(state, message)
    case 'error':
      return { scroll: false, error: message.message || '识别出错了' }
    default:
      return { scroll: false }
  }
}

function findEntry(state, segmentId) {
  return state.entries.find((item) => item.segment_id === segmentId)
}

function pushEntry(state, segmentId, text, status) {
  const entry = {
    segment_id: segmentId,
    raw_text: text,
    corrected_text: text,
    corrected_base: '',
    status
  }
  state.entries.push(entry)
  return entry
}

function handlePartial(state, message, useLLM) {
  // LLM 模式下已有未收尾的句子时，忽略临时预览，避免多出一个框
  if (useLLM && state.openId) {
    return { scroll: true }
  }

  const existing = findEntry(state, message.segment_id)
  if (existing) {
    existing.corrected_text = message.text
    existing.raw_text = message.text
  } else {
    pushEntry(state, message.segment_id, message.text, 'partial')
  }
  return { scroll: true }
}

function handleRaw(state, message, useLLM) {
  state.llmBuffer = ''

  if (useLLM && state.openId) {
    // 句子还没收尾：并进同一个框，先显示「已纠错 + 新片段原文」
    const entry = findEntry(state, state.openId)
    if (entry) {
      entry.corrected_base = entry.corrected_text
      entry.raw_text = (entry.raw_text + message.raw_text).trim()
      entry.corrected_text = (entry.corrected_base + message.raw_text).trim()
      entry.status = 'whisper_done'
    }
    state.latestSegId = message.segment_id
    return { scroll: true }
  }

  // 清掉残留的、属于别的片段的 partial
  const staleIndex = state.entries.findLastIndex(
    (item) => item.status === 'partial' && item.segment_id !== message.segment_id
  )
  if (staleIndex !== -1) {
    state.entries.splice(staleIndex, 1)
  }

  const nextStatus = useLLM ? 'whisper_done' : 'final'
  const existing = findEntry(state, message.segment_id)
  if (existing) {
    existing.raw_text = message.raw_text
    existing.corrected_text = message.raw_text
    existing.corrected_base = ''
    existing.status = nextStatus
  } else {
    pushEntry(state, message.segment_id, message.raw_text, nextStatus)
  }

  if (useLLM) {
    state.openId = message.segment_id
    state.latestSegId = message.segment_id
  }
  return { scroll: true }
}

function handleToken(state, message) {
  // 只收最新片段的 token，过期片段的流式响应当作没看见
  if (message.segment_id === state.latestSegId) {
    state.llmBuffer += message.token
  }
  return { scroll: false }
}

function handleLlmDone(state, message) {
  if (message.segment_id !== state.latestSegId) {
    return { scroll: false }
  }

  const entry = findEntry(state, state.openId)
  if (entry) {
    const fragment = (message.corrected_text || state.llmBuffer).trim()
    if (fragment) {
      entry.corrected_text = entry.corrected_base
        ? (entry.corrected_base + fragment).trim()
        : fragment
    }

    if (SENTENCE_END.test(entry.corrected_text)) {
      entry.status = 'final'
      state.openId = null
      state.latestSegId = null
    } else {
      entry.status = 'sentence_open'
    }
  }

  state.llmBuffer = ''
  return { scroll: true }
}

/** 停录音或关掉 LLM 时，把还开着的句子收尾。 */
export function closeOpenSentence(state) {
  if (!state.openId) return false

  const entry = findEntry(state, state.openId)
  if (entry) entry.status = 'final'

  state.openId = null
  state.latestSegId = null
  state.llmBuffer = ''
  return true
}
