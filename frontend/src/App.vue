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

      <div class="controls">
        <el-button 
          v-if="!isRecording"
          type="primary" 
          size="large"
          @click="startRecording"
          :disabled="!wsConnected"
        >
          <el-icon><Microphone /></el-icon> Start Speaking
        </el-button>
        
        <el-button 
          v-else
          type="danger" 
          size="large"
          @click="stopRecording"
        >
          <el-icon><VideoPause /></el-icon> Stop Speaking
        </el-button>
      </div>
      
      <div v-if="isRecording" class="recording-indicator">
        <span class="dot"></span> Recording in progress... (Sending audio chunks to backend)
      </div>

      <el-divider />

      <div class="transcription-area">
        <h3>Live Transcription</h3>
        <div class="transcript-box" ref="transcriptBox">
          <div v-if="transcriptHistory.length === 0" class="empty-state">
            Start recording to see live transcriptions...
          </div>
          <div v-else class="transcript-list">
             <div v-for="(item, index) in transcriptHistory" :key="index" class="transcript-item">
                <div class="corrected-text">{{ item.corrected_text }}</div>
                <div class="raw-text" v-if="item.raw_text !== item.corrected_text" title="Original Whisper Output">
                  (Raw: {{ item.raw_text }})
                </div>
             </div>
          </div>
        </div>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, nextTick } from 'vue'
import { Microphone, VideoPause } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

const wsConnected = ref(false)
const isRecording = ref(false)
const transcriptHistory = ref([])
const transcriptBox = ref(null)

let ws = null
let mediaRecorder = null
let audioContext = null
let processor = null
let source = null

// Initialize WebSocket connection to backend
const initWebSocket = () => {
  // Assuming frontend is essentially same host but different port, hardcoding for local dev
  const wsUrl = `ws://localhost:8000/api/ws/recognize`
  
  ws = new WebSocket(wsUrl)
  
  ws.onopen = () => {
    wsConnected.value = true
    console.log('WebSocket connected')
  }
  
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      if (data.type === 'transcription') {
        transcriptHistory.value.push({
          raw_text: data.raw_text,
          corrected_text: data.corrected_text
        })
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

const startRecording = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ 
      audio: {
        channelCount: 1, // Mono
        echoCancellation: true,
        noiseSuppression: true
      } 
    })
    
    // Use Web Audio API to capture raw PCM instead of compressed MediaRecorder formats
    // to bypass the need for FFmpeg on the backend.
    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 })
    source = audioContext.createMediaStreamSource(stream)
    
    // We want relatively large chunks for VAD processing (e.g. 512-4096 samples)
    // 4096 / 16000 is about 256ms chunk size. 
    processor = audioContext.createScriptProcessor(4096, 1, 1)
    
    source.connect(processor)
    processor.connect(audioContext.destination)
    
    processor.onaudioprocess = (e) => {
      if (!isRecording.value || !wsConnected.value) return
      
      const float32Array = e.inputBuffer.getChannelData(0)
      
      // Convert Float32Array (-1.0 to 1.0) to Int16Array (-32768 to 32767) for WebSocket transport
      // Backend expects Int16 PCM array.
      const int16Array = new Int16Array(float32Array.length)
      for (let i = 0; i < float32Array.length; i++) {
        let s = Math.max(-1, Math.min(1, float32Array[i]))
        int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF
      }
      
      if (ws && ws.readyState === WebSocket.OPEN) {
         ws.send(int16Array.buffer)
      }
    }
    
    // Keep a reference to stop later
    mediaRecorder = stream
    isRecording.value = true
    ElMessage.success('Microphone started')
    
  } catch (err) {
    console.error('Error accessing microphone:', err)
    ElMessage.error('Microphone access denied or not available')
  }
}

const stopRecording = () => {
  isRecording.value = false
  
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
  ElMessage.info('Recording stopped')
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
  margin: 30px 0;
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
