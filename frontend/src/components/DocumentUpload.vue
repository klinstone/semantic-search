<script setup>
import { ref } from 'vue'
import { api } from '@/api/client'
import { useNotification } from '@/composables/useNotification'
import { formatBytes } from '@/utils/format'

const emit = defineEmits(['uploaded'])
const notify = useNotification()

const MAX_SIZE_MB = 10
const MAX_SIZE = MAX_SIZE_MB * 1024 * 1024

// Accept attribute for the native file picker. Servers' magic-bytes check
// is the source of truth — this is just a UX hint.
const ACCEPT_ATTR =
  '.pdf,.txt,.docx,' +
  'application/pdf,' +
  'text/plain,' +
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document'

const ALLOWED_EXTENSIONS = ['.pdf', '.txt', '.docx']

const dragOver = ref(false)
const fileInput = ref(null)
const uploading = ref(false)
const progress = ref(0)
const currentFile = ref(null)

function pickFile() {
  if (!uploading.value) fileInput.value?.click()
}

function onFilePicked(e) {
  const file = e.target.files?.[0]
  if (file) handleFile(file)
  // Allow re-picking the same file
  e.target.value = ''
}

function onDrop(e) {
  dragOver.value = false
  if (uploading.value) return
  const file = e.dataTransfer?.files?.[0]
  if (file) handleFile(file)
}

function onDragOver() {
  if (!uploading.value) dragOver.value = true
}

function onDragLeave() {
  dragOver.value = false
}

function validate(file) {
  if (file.size === 0) return 'Файл пустой'
  if (file.size > MAX_SIZE) {
    return `Файл больше ${MAX_SIZE_MB} МБ (текущий ${formatBytes(file.size)})`
  }
  const name = file.name.toLowerCase()
  const okExt = ALLOWED_EXTENSIONS.some((ext) => name.endsWith(ext))
  if (!okExt) return 'Поддерживаются только PDF, TXT и DOCX'
  return null
}

async function handleFile(file) {
  const err = validate(file)
  if (err) {
    notify.warning(err)
    return
  }

  uploading.value = true
  currentFile.value = file
  progress.value = 0

  try {
    const doc = await api.uploadDocument(file, {
      onProgress: (p) => {
        progress.value = p
      }
    })
    emit('uploaded', doc)
  } catch (e) {
    notify.error(e.message || 'Не удалось загрузить файл')
  } finally {
    uploading.value = false
    currentFile.value = null
    progress.value = 0
  }
}
</script>

<template>
  <div
    class="upload-zone"
    :class="{ 'is-drag': dragOver, 'is-uploading': uploading }"
    role="button"
    tabindex="0"
    @click="pickFile"
    @keydown.enter.space.prevent="pickFile"
    @dragover.prevent="onDragOver"
    @dragenter.prevent="onDragOver"
    @dragleave.prevent="onDragLeave"
    @drop.prevent="onDrop"
  >
    <input
      ref="fileInput"
      type="file"
      :accept="ACCEPT_ATTR"
      class="hidden-input"
      @change="onFilePicked"
    />

    <template v-if="!uploading">
      <v-icon size="48" :color="dragOver ? 'primary' : 'grey-darken-1'">
        {{ dragOver ? 'mdi-file-arrow-up-down' : 'mdi-cloud-upload-outline' }}
      </v-icon>
      <div class="text-h6 mt-2">
        {{ dragOver ? 'Отпустите файл' : 'Перетащите файл или нажмите для выбора' }}
      </div>
      <div class="text-body-2 text-medium-emphasis mt-1">
        PDF, TXT, DOCX · до {{ MAX_SIZE_MB }} МБ
      </div>
    </template>

    <template v-else>
      <v-icon size="48" color="primary">mdi-cloud-sync</v-icon>
      <div class="text-h6 mt-2 text-truncate" style="max-width: 100%">
        {{ currentFile?.name }}
      </div>
      <div class="text-body-2 text-medium-emphasis mt-1">
        {{ formatBytes(currentFile?.size || 0) }} · {{ progress }}%
      </div>
      <v-progress-linear
        :model-value="progress"
        color="primary"
        height="6"
        rounded
        class="mt-3"
        style="max-width: 320px; margin-left: auto; margin-right: auto"
      />
    </template>
  </div>
</template>

<style scoped>
.upload-zone {
  border: 2px dashed rgba(0, 0, 0, 0.18);
  border-radius: 12px;
  background: #fafbfc;
  padding: 32px 24px;
  text-align: center;
  cursor: pointer;
  transition: border-color 0.15s ease, background-color 0.15s ease, transform 0.15s ease;
  outline: none;
}
.upload-zone:hover:not(.is-uploading),
.upload-zone:focus-visible:not(.is-uploading) {
  border-color: rgb(var(--v-theme-primary));
  background: #f3f5fc;
}
.upload-zone.is-drag {
  border-color: rgb(var(--v-theme-primary));
  background: #eef2ff;
  transform: scale(1.005);
}
.upload-zone.is-uploading {
  cursor: progress;
  border-style: solid;
  border-color: rgb(var(--v-theme-primary));
  background: #f3f5fc;
}
.hidden-input {
  display: none;
}
</style>
