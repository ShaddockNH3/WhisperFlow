<script setup>
import { useRecognition } from '@/composables/useRecognition'
import ControlPanel from '@/components/ControlPanel.vue'
import TranscriptPanel from '@/components/TranscriptPanel.vue'

// 录音与 WebSocket 在 composable 里，转写状态机在 services/transcript.js。
// 这里只摆版面。
const {
  wsConnected,
  isRecording,
  inputSource,
  useLLM,
  entries,
  scrollSignal,
  startRecording,
  stopRecording,
  setUseLLM
} = useRecognition()
</script>

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

      <control-panel
        v-model:input-source="inputSource"
        v-model:use-llm="useLLM"
        :ws-connected="wsConnected"
        :is-recording="isRecording"
        @start="startRecording"
        @stop="stopRecording"
        @llm-change="setUseLLM"
      />

      <el-divider />

      <transcript-panel :entries="entries" :scroll-signal="scrollSignal" />
    </el-card>
  </div>
</template>

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
</style>
