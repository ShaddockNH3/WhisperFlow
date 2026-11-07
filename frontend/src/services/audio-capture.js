/**
 * 麦克风 / 系统声音采集，把 Float32 转成 16kHz 单声道 Int16 帧往外送。
 * 只做采集，不认识 WebSocket，也不认识 Vue。
 */

const TARGET_SAMPLE_RATE = 16000

export const SOURCE = Object.freeze({
  MICROPHONE: 'microphone',
  SYSTEM: 'system'
})

/** 申请音频流。系统声音要用户在弹出的窗口里勾选「分享音频」。 */
export async function openStream(source) {
  if (source === SOURCE.MICROPHONE) {
    return navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true }
    })
  }

  // 多数浏览器不允许 video:false，拿到后立刻停掉视频轨
  const stream = await navigator.mediaDevices.getDisplayMedia({
    video: true,
    audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false }
  })
  stream.getVideoTracks().forEach((track) => track.stop())

  if (!stream.getAudioTracks().length) {
    stream.getTracks().forEach((track) => track.stop())
    throw new Error('NO_AUDIO_TRACK')
  }
  return stream
}

/** 重采样到 16k。浏览器采集的采样率不一定是 16k。 */
export async function resample(audioBuffer, targetSampleRate = TARGET_SAMPLE_RATE) {
  const offline = new OfflineAudioContext(
    audioBuffer.numberOfChannels,
    audioBuffer.duration * targetSampleRate,
    targetSampleRate
  )
  const bufferSource = offline.createBufferSource()
  bufferSource.buffer = audioBuffer
  bufferSource.connect(offline.destination)
  bufferSource.start(0)
  return offline.startRendering()
}

/**
 * 起采集链路：MediaStream → ScriptProcessor → Int16 帧 → onFrame。
 * 返回一个 stop 函数，调用方只管在合适的时机停。
 */
export function startCapture(stream, onFrame, { sampleRate = TARGET_SAMPLE_RATE, frameSize = 4096 } = {}) {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext
  const audioContext = new AudioContextClass({ sampleRate })
  const sourceNode = audioContext.createMediaStreamSource(stream)
  const processor = audioContext.createScriptProcessor(frameSize, 1, 1)

  sourceNode.connect(processor)
  processor.connect(audioContext.destination)

  processor.onaudioprocess = (event) => {
    const float32 = event.inputBuffer.getChannelData(0)
    const int16 = new Int16Array(float32.length)
    for (let index = 0; index < float32.length; index += 1) {
      const sample = Math.max(-1, Math.min(1, float32[index]))
      int16[index] = sample < 0 ? sample * 0x8000 : sample * 0x7fff
    }
    onFrame(int16.buffer)
  }

  return function stop() {
    processor.disconnect()
    processor.onaudioprocess = null
    sourceNode.disconnect()
    audioContext.close()
    stream.getTracks().forEach((track) => track.stop())
  }
}
