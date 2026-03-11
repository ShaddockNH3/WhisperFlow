<template>
  <div class="app-container">
    <el-card class="box-card">
      <template #header>
        <div class="card-header">
          <span>WhisperFlow Real-time Recognition</span>
          <el-tag :type="wsConnected ? 'success' : 'danger'">
            {{ wsConnected ? 'Connected' : 'Disconnected' }}
          </el-tag>
        </div>
      </template>

      <div class="source-selector">
        <el-radio-group v-model="inputSource" :disabled="isRecording" size="default">
          <el-radio-button value="microphone">
            <el-icon style="margin-right:4px"><Microphone /></el-icon> 麦克风
          </el-radio-button>
          <el-radio-button value="system">
            <el-icon style="margin-right:4px"><Monitor /></el-icon> 系统声音
          </el-radio-button>
        </el-radio-group>

        <el-divider direction="vertical" />

        <el-tooltip content="开启后 Whisper 识别的原始文本将由 GLM-4-Flash 进行语义纠错" placement="bottom">
          <el-switch
            v-model="useLLM"
            active-text="AI 纠错"
            inactive-text="仅识别"
            @change="onUseLLMChange"
          />
        </el-tooltip>
      </div>

      <div class="controls">
        <el-button 
          v-if="!isRecording"
          type="primary" 
          size="large"
          @click="startRecording"
          :disabled="!wsConnected"
        >
          <el-icon><Microphone /></el-icon> 开始识别
        </el-button>
        
        <el-button 
          v-else
          type="danger" 
          size="large"
          @click="stopRecording"
        >
          <el-icon><VideoPause /></el-icon> 停止识别
        </el-button>
      </div>
      
      <div v-if="isRecording" class="recording-indicator">
        <span class="dot"></span>
        <template v-if="inputSource === 'microphone'">麦克风录音中...</template>
        <template v-else>系统声音捕获中...<span class="source-tip">（请在弹窗中勾选"分享音频"）</span></template>
      </div>

      <el-divider />

      <div class="transcription-area">
        <h3>Live Transcription</h3>
        <div class="transcript-box" ref="transcriptBox">
          <div v-if="transcriptHistory.length === 0 && !currentPartial" class="empty-state">
            开始录音后此处将显示识别结果...
          </div>
          <div v-else class="transcript-list">
             <div v-for="(item, index) in transcriptHistory" :key="index" class="transcript-item"
               :class="{ 'transcript-item--streaming': item.streaming }">
                <!-- 流式期间：只显示正在生成的 LLM 文本（从空开始逐字填入），不展示原始行 -->
                <!-- 完成后：显示纠错文本，若与原始不同则在下方显示原始供参考 -->
                <div class="corrected-text">
                  {{ item.corrected_text }}<span v-if="item.streaming" class="cursor">|</span>
                </div>
                <div class="raw-text"
                  v-if="!item.usedLLM && !item.streaming && item.corrected_text !== item.raw_text"
                  title="Whisper 原始识别">
                  原始：{{ item.raw_text }}
                </div>
             </div>
             <!-- 语音预览行（VAD 缓冲中） -->
             <div v-if="currentPartial" class="transcript-item transcript-item--partial">
               <div class="corrected-text">{{ currentPartial }}<span class="cursor">|</span></div>
             </div>
          </div>
        </div>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import { Microphone, VideoPause, Monitor } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

const wsConnected = ref(false)
const isRecording = ref(false)
const inputSource = ref('microphone')
const useLLM = ref(true)
const transcriptHistory = ref([])
const currentPartial = ref('')
const transcriptBox = ref(null)
// Accumulates LLM tokens silently until llm_done, then replaces the Whisper text
const llmBuffer = ref('')

let ws = null
let mediaRecorder = null
let audioContext = null
let processor = null
let source = null

// Initialize WebSocket connection to backend
const initWebSocket = () => {
  // Use the current window's hostname so LAN devices can connect
  const wsUrl = `ws://${window.location.hostname}:8000/api/ws/recognize`
  
  ws = new WebSocket(wsUrl)
  
  ws.onopen = () => {
    wsConnected.value = true
    console.log('WebSocket connected')
  }
  
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)

      if (data.type === 'transcription_raw') {
        currentPartial.value = ''
        llmBuffer.value = ''
        // 立即显示 Whisper 输出；若开启 LLM 则保持 streaming 光标表示正在纠错
        transcriptHistory.value.push({
          raw_text: data.raw_text,
          corrected_text: data.raw_text,
          streaming: useLLM.value,
          usedLLM: false
        })
        scrollToBottom()

      } else if (data.type === 'llm_token') {
        // 静默累积，不更新显示，等 llm_done 后一次性替换
        llmBuffer.value += data.token

      } else if (data.type === 'llm_done') {
        const last = transcriptHistory.value[transcriptHistory.value.length - 1]
        if (last && last.streaming) {
          if (llmBuffer.value.trim()) {
            // 用完整的 LLM 输出一次性覆盖 Whisper 文本
            last.corrected_text = llmBuffer.value.trim()
            last.usedLLM = true
          }
          last.streaming = false
          llmBuffer.value = ''
          scrollToBottom()
        }

      } else if (data.type === 'partial') {
        currentPartial.value = data.text
        scrollToBottom()

      } else if (data.type === 'error') {
        ElMessage.error(data.message)
      }
    } catch(e) {
      console.error('Error parsing WS message:', e)
    }
  }
  
  ws.onclose = () => {
    wsConnected.value = false
    console.log('WebSocket disconnected')
    if (isRecording.value) {
        stopRecording()
    }
    // Attempt reconnect after delay
    setTimeout(initWebSocket, 3000)
  }
  
  ws.onerror = (error) => {
    console.error('WebSocket Error', error)
  }
}

// Function to resample AudioBuffer to 16kHz
const resampleAudioBuffer = async (audioBuffer, targetSampleRate) => {
    const offlineCtx = new OfflineAudioContext(
        audioBuffer.numberOfChannels,
        audioBuffer.duration * targetSampleRate,
        targetSampleRate
    );
    const bufferSource = offlineCtx.createBufferSource();
    bufferSource.buffer = audioBuffer;
    bufferSource.connect(offlineCtx.destination);
    bufferSource.start(0);
    return await offlineCtx.startRendering();
}

const startAudioProcessing = (stream) => {
  audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 })
  source = audioContext.createMediaStreamSource(stream)
  processor = audioContext.createScriptProcessor(4096, 1, 1)

  source.connect(processor)
  processor.connect(audioContext.destination)

  processor.onaudioprocess = (e) => {
    if (!isRecording.value || !wsConnected.value) return
    const float32Array = e.inputBuffer.getChannelData(0)
    const int16Array = new Int16Array(float32Array.length)
    for (let i = 0; i < float32Array.length; i++) {
      const s = Math.max(-1, Math.min(1, float32Array[i]))
      int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF
    }
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(int16Array.buffer)
    }
  }

  // 监听系统声音流结束（用户在弹窗中点了停止共享）
  stream.getAudioTracks().forEach(track => {
    track.onended = () => {
      if (isRecording.value) stopRecording()
    }
  })

  mediaRecorder = stream
  isRecording.value = true
}

const startRecording = async () => {
  try {
    if (inputSource.value === 'microphone') {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        }
      })
      startAudioProcessing(stream)
      ElMessage.success('麦克风启动成功')
    } else {
      // 捕获系统/标签页声音，Chrome 下需在弹窗中手动勾选"分享音频"
      // 注：大多数浏览器要求 video 不能为 false，获取后立即停止视频轨道
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false
        }
      })
      // 立即停止视频轨道，只保留音频
      stream.getVideoTracks().forEach(t => t.stop())
      if (!stream.getAudioTracks().length) {
        stream.getTracks().forEach(t => t.stop())
        ElMessage.warning('未检测到音频轨道，请在弹窗中勾选"分享音频"后重试')
        return
      }
      startAudioProcessing(stream)
      ElMessage.success('系统声音捕获启动成功')
    }
  } catch (err) {
    if (err.name === 'NotAllowedError') {
      ElMessage.error('用户取消或拒绝了权限请求')
    } else {
      console.error('Error starting recording:', err)
      ElMessage.error('启动失败：' + err.message)
    }
  }
}

const stopRecording = () => {
  isRecording.value = false
  currentPartial.value = ''
  
  if (processor) {
    processor.disconnect()
    processor.onaudioprocess = null
  }
  if (source) {
    source.disconnect()
  }
  if (audioContext) {
    audioContext.close()
  }
  
  if (mediaRecorder) {
    mediaRecorder.getTracks().forEach(track => track.stop())
    mediaRecorder = null
  }
  ElMessage.info('识别已停止')
}

const onUseLLMChange = (val) => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'set_use_llm', value: val }))
  }
}

const scrollToBottom = () => {
  nextTick(() => {
    if (transcriptBox.value) {
      transcriptBox.value.scrollTop = transcriptBox.value.scrollHeight
    }
  })
}

onMounted(() => {
  initWebSocket()
})

onUnmounted(() => {
  stopRecording()
  if (ws) {
    ws.close()
  }
})
</script>

<style scoped>
.app-container {
  display: flex;
  justify-content: center;
  padding: 40px;
  background-color: #f5f7fa;
  min-height: 100vh;
}

.box-card {
  width: 100%;
  max-width: 800px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: bold;
  font-size: 1.2rem;
}

.controls {
  display: flex;
  justify-content: center;
  margin: 20px 0;
}

.source-selector {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 12px;
  margin: 20px 0 0;
}

.recording-indicator {
  text-align: center;
  color: #e6a23c;
  font-size: 0.9rem;
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  flex-wrap: wrap;
}

.source-tip {
  color: #909399;
  font-size: 0.82rem;
}

.dot {
  width: 10px;
  height: 10px;
  background-color: #f56c6c;
  border-radius: 50%;
  animation: pulse 1.5s infinite;
}

@keyframes pulse {
  0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(245, 108, 108, 0.7); }
  70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(245, 108, 108, 0); }
  100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(245, 108, 108, 0); }
}

.transcription-area h3 {
  margin-top: 0;
  color: #303133;
}

.transcript-box {
  background-color: #fafafa;
  border: 1px solid #ebeef5;
  border-radius: 8px;
  min-height: 300px;
  max-height: 500px;
  overflow-y: auto;
  padding: 20px;
}

.empty-state {
  color: #909399;
  text-align: center;
  margin-top: 100px;
  font-style: italic;
}

.transcript-list {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.transcript-item {
  background-color: white;
  padding: 15px;
  border-radius: 6px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
  border-left: 4px solid #409eff;
}

.transcript-item--partial {
  border-left-color: #e6a23c;
  opacity: 0.85;
}

.transcript-item--streaming {
  border-left-color: #67c23a;
}

.waiting-indicator {
  color: #909399;
  font-style: italic;
  font-size: 0.9rem;
}

.cursor {
  display: inline-block;
  margin-left: 1px;
  color: #409eff;
  animation: blink 1s step-start infinite;
}

@keyframes blink {
  50% { opacity: 0; }
}

.corrected-text {
  font-size: 1.1rem;
  color: #303133;
  line-height: 1.5;
}

.raw-text {
  font-size: 0.85rem;
  color: #909399;
  margin-top: 5px;
  font-style: italic;
}
</style>
