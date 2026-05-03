<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { api } from '@/api/client'
import { useNotification } from '@/composables/useNotification'
import {
  formatBytes,
  formatDate,
  STATUS_LABELS,
  STATUS_COLORS,
  isTerminalStatus,
  mimeIcon
} from '@/utils/format'
import DocumentUpload from '@/components/DocumentUpload.vue'

const notify = useNotification()

const items = ref([])
const total = ref(0)
const limit = ref(20)
const page = ref(1)
const statusFilter = ref(null)
const loading = ref(false)

const deleteTarget = ref(null)
const deleting = ref(false)

const offset = computed(() => (page.value - 1) * limit.value)
const pageCount = computed(() => Math.max(1, Math.ceil(total.value / limit.value)))

const hasPending = computed(() =>
  items.value.some((d) => !isTerminalStatus(d.status))
)

const headers = [
  { title: 'Имя файла', key: 'filename', sortable: false },
  { title: 'Размер', key: 'size_bytes', sortable: false, width: 110 },
  { title: 'Статус', key: 'status', sortable: false, width: 170 },
  { title: 'Чанков', key: 'chunks_count', sortable: false, width: 100, align: 'end' },
  { title: 'Загружен', key: 'uploaded_at', sortable: false, width: 170 },
  { title: '', key: 'actions', sortable: false, width: 60, align: 'end' }
]

const statusOptions = [
  { value: null, title: 'Все статусы' },
  { value: 'pending', title: STATUS_LABELS.pending },
  { value: 'processing', title: STATUS_LABELS.processing },
  { value: 'indexed', title: STATUS_LABELS.indexed },
  { value: 'failed', title: STATUS_LABELS.failed }
]

let pollHandle = null
const POLL_INTERVAL_MS = 2500

async function load({ silent = false } = {}) {
  if (!silent) loading.value = true
  try {
    const res = await api.listDocuments({
      limit: limit.value,
      offset: offset.value,
      status: statusFilter.value
    })
    items.value = res.items
    total.value = res.total
  } catch (e) {
    notify.error(e.message || 'Не удалось загрузить список документов')
  } finally {
    loading.value = false
  }
}

function startPolling() {
  stopPolling()
  pollHandle = setInterval(() => {
    if (hasPending.value && !loading.value) {
      load({ silent: true })
    }
  }, POLL_INTERVAL_MS)
}

function stopPolling() {
  if (pollHandle) {
    clearInterval(pollHandle)
    pollHandle = null
  }
}

onMounted(() => {
  load()
  startPolling()
})

onUnmounted(stopPolling)

// Reset to page 1 when filter or page size changes; the second watcher then loads.
watch([limit, statusFilter], () => {
  if (page.value !== 1) {
    page.value = 1
  } else {
    load()
  }
})
watch(page, () => load())

async function onUploaded(doc) {
  notify.success(`Файл «${doc.filename}» поставлен в очередь индексации`)
  // Jump to the first page so the new document is visible.
  if (page.value !== 1) {
    page.value = 1
  } else {
    await load({ silent: true })
  }
}

function askDelete(doc) {
  deleteTarget.value = doc
}

async function confirmDelete() {
  if (!deleteTarget.value) return
  deleting.value = true
  try {
    await api.deleteDocument(deleteTarget.value.id)
    notify.success('Документ удалён')
    deleteTarget.value = null
    // If we just deleted the last item on a non-first page, step back.
    if (items.value.length === 1 && page.value > 1) {
      page.value -= 1
    } else {
      await load({ silent: true })
    }
  } catch (e) {
    notify.error(e.message || 'Не удалось удалить документ')
  } finally {
    deleting.value = false
  }
}
</script>

<template>
  <div>
    <div class="d-flex align-center justify-space-between mb-4 mt-2 flex-wrap header-row">
      <h1 class="text-h4 font-weight-bold">Документы</h1>
      <div class="text-body-2 text-medium-emphasis">
        Всего: <strong>{{ total }}</strong>
      </div>
    </div>

    <DocumentUpload class="mb-6" @uploaded="onUploaded" />

    <v-card elevation="2">
      <v-toolbar density="compact" color="transparent" flat class="px-2">
        <v-toolbar-title class="text-subtitle-1">Список документов</v-toolbar-title>
        <v-spacer />
        <v-select
          v-model="statusFilter"
          :items="statusOptions"
          item-title="title"
          item-value="value"
          density="compact"
          hide-details
          style="max-width: 220px"
        />
        <v-btn
          icon="mdi-refresh"
          variant="text"
          :loading="loading"
          class="ml-2"
          @click="load()"
        />
      </v-toolbar>

      <v-divider />

      <v-data-table
        :headers="headers"
        :items="items"
        :loading="loading"
        :items-per-page="limit"
        hide-default-footer
        class="documents-table"
        no-data-text="Документы не загружены"
        loading-text="Загрузка..."
      >
        <template #item.filename="{ item }">
          <div class="cell-filename">
            <v-icon size="small" color="grey-darken-1">{{ mimeIcon(item.mime_type) }}</v-icon>
            <span class="text-truncate" :title="item.filename">{{ item.filename }}</span>
          </div>
        </template>

        <template #item.size_bytes="{ item }">
          {{ formatBytes(item.size_bytes) }}
        </template>

        <template #item.status="{ item }">
          <v-chip
            :color="STATUS_COLORS[item.status] || 'grey'"
            size="small"
            variant="tonal"
          >
            <v-progress-circular
              v-if="!isTerminalStatus(item.status)"
              indeterminate
              size="12"
              width="2"
              class="mr-1"
            />
            {{ STATUS_LABELS[item.status] || item.status }}
          </v-chip>
        </template>

        <template #item.chunks_count="{ item }">
          <span class="text-body-2">{{ item.chunks_count ?? '—' }}</span>
        </template>

        <template #item.uploaded_at="{ item }">
          <span class="text-body-2">{{ formatDate(item.uploaded_at) }}</span>
        </template>

        <template #item.actions="{ item }">
          <v-btn
            icon="mdi-delete-outline"
            variant="text"
            density="comfortable"
            color="error"
            size="small"
            @click="askDelete(item)"
          />
        </template>
      </v-data-table>

      <v-divider />

      <div class="d-flex align-center pa-3 footer-row">
        <v-select
          v-model="limit"
          :items="[10, 20, 50, 100]"
          label="На странице"
          density="compact"
          hide-details
          style="max-width: 140px"
        />
        <v-spacer />
        <v-pagination
          v-model="page"
          :length="pageCount"
          :total-visible="5"
          density="comfortable"
        />
      </div>
    </v-card>

    <v-dialog
      :model-value="!!deleteTarget"
      max-width="480"
      :persistent="deleting"
      @update:model-value="(v) => !v && (deleteTarget = null)"
    >
      <v-card>
        <v-card-title>Удалить документ?</v-card-title>
        <v-card-text>
          Документ <strong>{{ deleteTarget?.filename }}</strong> будет полностью удалён
          вместе со всеми связанными чанками. Это действие нельзя отменить.
        </v-card-text>
        <v-card-actions>
          <v-spacer />
          <v-btn variant="text" :disabled="deleting" @click="deleteTarget = null">
            Отмена
          </v-btn>
          <v-btn color="error" :loading="deleting" @click="confirmDelete">
            Удалить
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </div>
</template>

<style scoped>
.header-row {
  gap: 12px;
}
.cell-filename {
  display: flex;
  gap: 8px;
  align-items: center;
  min-width: 0;
}
.cell-filename > span {
  min-width: 0;
}
.footer-row {
  gap: 12px;
  flex-wrap: wrap;
}
.documents-table :deep(td) {
  font-size: 14px;
}
</style>
