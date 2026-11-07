import { computed, onMounted, onUnmounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { applyMessage, closeOpenSentence, createTranscriptState } from '@/services/transcript'
import { SOURCE, openStream, startCapture } from '@/services/audio-capture'

const WS_PORT = 8000
const RECONNECT_DELAY = 3000

/**
 * 录音 + WebSocket 的接线层：负责建立连接、采集音频、把消息塞给状态机。
 *
 * 状态机在 services/transcript.js（纯函数，可单测），采集在 services/audio-capture.js。
 * 这里只剩下"什么时候调谁"。
 */
export function useRecognition() {
  const wsConnected = ref(false)
  const isRecording = ref(false)
  const inputSource = ref(SOURCE.MICROPHONE)
  const useLLM = ref(true)

  const state = reactive(createTranscriptState())
  const entries = computed(() => state.entries)
  const scrollSignal = ref(0)

  let ws = null
  let stopCapture = null
  let disposed = false
  let reconnectTimer = null

  function scrollToBottom() {
    scrollSignal.value += 1
  }

  function connect() {
    const url = `ws://${window.location.hostname}:${WS_PORT}/api/ws/recognize`
    ws = new WebSocket(url)

    ws.onopen = () => {
      wsConnected.value = true
    }

    ws.onmessage = (event) => {
      let message
      try {
        message = JSON.parse(event.data)
      } catch (error) {
        console.error('WebSocket 消息解析失败:', error)
        return
      }

      const result = applyMessage(state, message, { useLLM: useLLM.value })
      if (result.error) ElMessage.error(result.error)
      if (result.scroll) scrollToBottom()
    }

    ws.onclose = () => {
      wsConnected.value = false
      if (isRecording.value) stopRecording()

      if (!disposed) {
        reconnectTimer = setTimeout(connect, RECONNECT_DELAY)
      }
    }

    ws.onerror = (error) => {
      console.error('WebSocket 错误:', error)
    }
  }

  async function startRecording() {
    try {
      const stream = await openStream(inputSource.value)

      stopCapture = startCapture(stream, (frame) => {
        if (!isRecording.value || !wsConnected.value) return
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(frame)
        }
      })

      // 用户在弹出的共享窗口里点了停止
      stream.getAudioTracks().forEach((track) => {
        track.onended = () => {
          if (isRecording.value) stopRecording()
        }
      })

      isRecording.value = true
      ElMessage.success(inputSource.value === SOURCE.MICROPHONE ? '麦克风启动成功' : '系统声音捕获启动成功')
    } catch (error) {
      if (error?.name === 'NotAllowedError') {
        ElMessage.error('用户取消或拒绝了权限请求')
      } else if (error?.message === 'NO_AUDIO_TRACK') {
        ElMessage.warning('未检测到音频轨道，请在弹窗中勾选「分享音频」后重试')
      } else {
        console.error('启动录音失败:', error)
        ElMessage.error(`启动失败：${error?.message || error}`)
      }
    }
  }

  function stopRecording() {
    isRecording.value = false

    if (closeOpenSentence(state)) {
      scrollToBottom()
    }

    if (stopCapture) {
      stopCapture()
      stopCapture = null
    }
    ElMessage.info('识别已停止')
  }

  /** 切换 AI 纠错：通知后端；关掉时把当前句子收尾。 */
  function setUseLLM(value) {
    useLLM.value = value

    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'set_use_llm', value }))
    }

    if (!value && closeOpenSentence(state)) {
      scrollToBottom()
    }
  }

  onMounted(connect)

  onUnmounted(() => {
    disposed = true
    if (reconnectTimer) clearTimeout(reconnectTimer)
    stopRecording()
    if (ws) ws.close()
  })

  return {
    wsConnected,
    isRecording,
    inputSource,
    useLLM,
    entries,
    scrollSignal,
    startRecording,
    stopRecording,
    setUseLLM,
    scrollToBottom
  }
}
