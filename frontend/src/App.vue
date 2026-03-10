<template>
  <div class="app-container">
    <el-card class="box-card">
      <template #header>
        <div class="card-header">
          <span>WhisperFlow Audio Recognition</span>
        </div>
      </template>
      
      <div class="upload-section">
        <el-upload
          class="upload-demo"
          drag
          action="#"
          :auto-upload="false"
          :on-change="handleFileChange"
          :show-file-list="false"
          accept="audio/*"
        >
          <el-icon class="el-icon--upload"><upload-filled /></el-icon>
          <div class="el-upload__text">
            Drop audio file here or <em>click to upload</em>
          </div>
        </el-upload>
      </div>

      <div class="actions" v-if="selectedFile">
        <div class="file-info">Selected: {{ selectedFile.name }}</div>
        <el-button type="primary" :loading="isRecognizing" @click="handleRecognize">
          Recognize Audio
        </el-button>
        <el-button @click="selectedFile = null" :disabled="isRecognizing">
          Clear
        </el-button>
      </div>

      <el-divider v-if="results.length > 0" />

      <div class="results-section" v-if="results.length > 0">
        <h3>Recognition Results</h3>
        <el-table :data="results" style="width: 100%" v-loading="isDeleting">
          <el-table-column prop="filename" label="File Name" width="180" />
          <el-table-column prop="transcription" label="Transcription" />
          <el-table-column prop="status" label="Status" width="100">
            <template #default="scope">
              <el-tag :type="scope.row.status === 'completed' ? 'success' : 'info'">
                {{ scope.row.status }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column fixed="right" label="Operations" width="120">
            <template #default="scope">
              <el-button link type="danger" size="small" @click="handleDelete(scope.row.task_id)">
                Delete
              </el-button>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { UploadFilled } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { recognizeAudio, deleteRecord } from './services/api.js'

const selectedFile = ref(null)
const isRecognizing = ref(false)
const isDeleting = ref(false)
const results = ref([])

const handleFileChange = (file) => {
  if (file.raw.type.startsWith('audio/')) {
    selectedFile.value = file.raw
  } else {
    ElMessage.error('Please select an audio file!')
  }
}

const handleRecognize = async () => {
  if (!selectedFile.value) return
  
  isRecognizing.value = true
  try {
    const data = await recognizeAudio(selectedFile.value)
    results.value.unshift(data)
    ElMessage.success('Audio recognized successfully')
    selectedFile.value = null
  } catch (error) {
    ElMessage.error(error.response?.data?.detail || 'Failed to recognize audio')
  } finally {
    isRecognizing.value = false
  }
}

const handleDelete = async (taskId) => {
  isDeleting.value = true
  try {
    await deleteRecord(taskId)
    results.value = results.value.filter(r => r.task_id !== taskId)
    ElMessage.success('Record deleted successfully')
  } catch (error) {
    ElMessage.error(error.response?.data?.detail || 'Failed to delete record')
  } finally {
    isDeleting.value = false
  }
}
</script>

<style scoped>
.app-container {
  display: flex;
  justify-content: center;
  padding: 20px;
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

.upload-section {
  margin-bottom: 20px;
}

.actions {
  display: flex;
  align-items: center;
  gap: 15px;
  margin-bottom: 20px;
}

.file-info {
  flex-grow: 1;
  color: #606266;
  font-size: 0.9rem;
}

.results-section {
  margin-top: 20px;
}

.results-section h3 {
  margin-bottom: 15px;
  color: #303133;
}
</style>
