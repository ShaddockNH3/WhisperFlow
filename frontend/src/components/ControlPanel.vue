<script setup>
import { Microphone, VideoPause, Monitor } from '@element-plus/icons-vue'
import { SOURCE } from '@/services/audio-capture'

/** 音源选择 + AI 纠错开关 + 开始/停止按钮 + 录音提示。 */
const props = defineProps({
  wsConnected: { type: Boolean, default: false },
  isRecording: { type: Boolean, default: false },
  inputSource: { type: String, default: SOURCE.MICROPHONE },
  useLLM: { type: Boolean, default: true }
})

const emit = defineEmits(['update:inputSource', 'update:useLLM', 'start', 'stop', 'llm-change'])
</script>

<template>
  <div class="source-selector">
    <el-radio-group
      :model-value="props.inputSource"
      :disabled="props.isRecording"
      size="default"
      @update:model-value="(value) => emit('update:inputSource', value)"
    >
      <el-radio-button :value="SOURCE.MICROPHONE">
        <el-icon class="radio-icon"><Microphone /></el-icon> 麦克风
      </el-radio-button>
      <el-radio-button :value="SOURCE.SYSTEM">
        <el-icon class="radio-icon"><Monitor /></el-icon> 系统声音
      </el-radio-button>
    </el-radio-group>

    <el-divider direction="vertical" />

    <el-tooltip
      content="开启后 Whisper 识别的原始文本将由 GLM-4-Flash 进行语义纠错"
      placement="bottom"
    >
      <el-switch
        :model-value="props.useLLM"
        active-text="AI 纠错"
        inactive-text="仅识别"
        @update:model-value="(value) => { emit('update:useLLM', value); emit('llm-change', value) }"
      />
    </el-tooltip>
  </div>

  <div class="controls">
    <el-button
      v-if="!props.isRecording"
      type="primary"
      size="large"
      :disabled="!props.wsConnected"
      @click="emit('start')"
    >
      <el-icon><Microphone /></el-icon> 开始识别
    </el-button>

    <el-button v-else type="danger" size="large" @click="emit('stop')">
      <el-icon><VideoPause /></el-icon> 停止识别
    </el-button>
  </div>

  <div v-if="props.isRecording" class="recording-indicator">
    <span class="dot"></span>
    <template v-if="props.inputSource === SOURCE.MICROPHONE">麦克风录音中...</template>
    <template v-else>
      系统声音捕获中...<span class="source-tip">（请在弹窗中勾选「分享音频」）</span>
    </template>
  </div>
</template>

<style scoped>
.source-selector {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 12px;
  margin: 20px 0 0;
}

.radio-icon {
  margin-right: 4px;
}

.controls {
  display: flex;
  justify-content: center;
  margin: 20px 0;
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
  0% {
    transform: scale(0.95);
    box-shadow: 0 0 0 0 rgba(245, 108, 108, 0.7);
  }
  70% {
    transform: scale(1);
    box-shadow: 0 0 0 10px rgba(245, 108, 108, 0);
  }
  100% {
    transform: scale(0.95);
    box-shadow: 0 0 0 0 rgba(245, 108, 108, 0);
  }
}
</style>
