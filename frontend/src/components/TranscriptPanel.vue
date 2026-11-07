<script setup>
import { nextTick, ref, watch } from 'vue'

/**
 * 转写结果面板。条目状态：partial（预览）/ whisper_done（等纠错）
 * / sentence_open（句子未收尾）/ final（定稿）。
 */
const props = defineProps({
  entries: { type: Array, default: () => [] },
  scrollSignal: { type: Number, default: 0 }
})

const box = ref(null)

watch(
  () => props.scrollSignal,
  () => {
    nextTick(() => {
      if (box.value) box.value.scrollTop = box.value.scrollHeight
    })
  }
)

function itemClass(item) {
  return {
    'transcript-item--partial': item.status === 'partial',
    'transcript-item--pending': item.status === 'whisper_done',
    'transcript-item--final-llm': item.status === 'final' && item.raw_text !== item.corrected_text,
    'transcript-item--sentence-open': item.status === 'sentence_open'
  }
}

function textClass(item) {
  return {
    'text--gray': item.status === 'partial' || item.status === 'whisper_done',
    'text--underline': item.status === 'partial'
  }
}
</script>

<template>
  <div class="transcription-area">
    <h3>Live Transcription</h3>

    <div ref="box" class="transcript-box">
      <div v-if="props.entries.length === 0" class="empty-state">
        开始录音后此处将显示识别结果...
      </div>

      <div v-else class="transcript-list">
        <div
          v-for="item in props.entries"
          :key="item.segment_id"
          class="transcript-item"
          :class="itemClass(item)"
        >
          <div class="corrected-text" :class="textClass(item)">
            {{ item.corrected_text }}<span v-if="item.status === 'partial'" class="cursor">|</span>
          </div>

          <div
            v-if="item.status === 'final' && item.raw_text !== item.corrected_text"
            class="raw-text"
            title="Whisper 原始识别"
          >
            原始：{{ item.raw_text }}
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
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
  opacity: 0.9;
}

.transcript-item--pending {
  border-left-color: #909399;
  opacity: 0.85;
}

.transcript-item--final-llm {
  border-left-color: #67c23a;
}

/* LLM 判定句子还没说完，用虚线提醒 */
.transcript-item--sentence-open {
  border-left-color: #e6a23c;
  border-left-style: dashed;
}

.text--gray {
  color: #909399;
}

.text--underline {
  text-decoration: underline dotted #e6a23c;
  text-underline-offset: 3px;
}

.cursor {
  display: inline-block;
  margin-left: 1px;
  color: #409eff;
  animation: blink 1s step-start infinite;
}

@keyframes blink {
  50% {
    opacity: 0;
  }
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
