<script setup>
import { ref, computed } from 'vue'
import { api } from '@/api/client'
import { formatBytes, formatDate, mimeIcon } from '@/utils/format'

const props = defineProps({
  // null = no filter (search across all indexed documents)
  // string[] = restrict search to these document UUIDs
  modelValue: { type: Array, default: null }
})
const emit = defineEmits(['update:modelValue'])

const open = ref(false)
const docs = ref([])
const total = ref(0)
const initialized = ref(false)
const filter = ref('')
const draft = ref(new Set())

const isFiltered = computed(() => Array.isArray(props.modelValue))
const selectedCount = computed(() => props.modelValue?.length ?? 0)

const filteredDocs = computed(() => {
  const q = filter.value.trim().toLowerCase()
  return q ? docs.value.filter(d => d.filename.toLowerCase().includes(q)) : docs.value
})
const allFilteredSelected = computed(() =>
  filteredDocs.value.length > 0 && filteredDocs.value.every(d => draft.value.has(d.id))
)
const someFilteredSelected = computed(() =>
  filteredDocs.value.some(d => draft.value.has(d.id))
)

async function loadIndexed() {
  try {
    // 100 covers MVP-scale corpora; if it grows past that, the filter input
    // inside the dialog is the way to find a specific document.
    const res = await api.listDocuments({ status: 'indexed', limit: 100 })
    docs.value = res.items
    total.value = res.total
  } catch {
    docs.value = []
    total.value = 0
  } finally {
    initialized.value = true
  }
}

async function openDialog() {
  await loadIndexed()
  // Drop UUIDs that no longer correspond to any indexed document
  // (e.g. someone deleted them in another tab while we kept them selected).
  const valid = new Set(docs.value.map(d => d.id))
  draft.value = new Set((props.modelValue ?? []).filter(id => valid.has(id)))
  filter.value = ''
  open.value = true
}

function toggle(id) {
  const next = new Set(draft.value)
  next.has(id) ? next.delete(id) : next.add(id)
  draft.value = next
}

function toggleAllFiltered() {
  const next = new Set(draft.value)
  const op = allFilteredSelected.value ? 'delete' : 'add'
  filteredDocs.value.forEach(d => next[op](d.id))
  draft.value = next
}

function apply() {
  // Empty draft is treated as "no filter" — applying nothing is the same
  // user intent as resetting. Avoids the [] = "search nothing" backend
  // semantics surfacing through the UI.
  emit('update:modelValue', draft.value.size === 0 ? null : [...draft.value])
  open.value = false
}

function reset() {
  emit('update:modelValue', null)
  draft.value = new Set()
  open.value = false
}

// Load once on mount so the chip can show the corpus size immediately.
loadIndexed()
</script>

<template>
  <div class="d-flex align-center" style="gap: 4px">
    <v-chip
      :color="isFiltered ? 'primary' : undefined"
      :variant="isFiltered ? 'flat' : 'tonal'"
      :disabled="!initialized || total === 0"
      size="small"
      class="scope-chip"
      @click="openDialog"
    >
      <v-icon start size="small">mdi-filter-variant</v-icon>
      <template v-if="!initialized">Загрузка…</template>
      <template v-else-if="total === 0">Нет готовых документов</template>
      <template v-else-if="!isFiltered">Все документы ({{ total }})</template>
      <template v-else>Выбрано {{ selectedCount }} из {{ total }}</template>
      <v-icon v-if="initialized && total > 0" end size="x-small">mdi-chevron-down</v-icon>
    </v-chip>
    <v-btn
      v-if="isFiltered"
      icon="mdi-close"
      size="x-small"
      variant="text"
      title="Сбросить выбор"
      @click="reset"
    />

    <v-dialog v-model="open" max-width="640">
      <v-card>
        <v-card-title>Документы для поиска</v-card-title>
        <v-divider />
        <div class="pa-3">
          <v-text-field
            v-model="filter"
            placeholder="Фильтр по имени файла"
            prepend-inner-icon="mdi-magnify"
            density="compact"
            hide-details
            clearable
          />
        </div>
        <v-divider />
        <div class="scope-list">
          <v-list density="compact" lines="two">
            <v-list-item
              v-if="filteredDocs.length > 0"
              :title="allFilteredSelected ? 'Снять выбор со всех' : 'Выбрать все'"
              @click="toggleAllFiltered"
            >
              <template #prepend>
                <v-checkbox-btn
                  :model-value="allFilteredSelected"
                  :indeterminate="!allFilteredSelected && someFilteredSelected"
                  @click.stop="toggleAllFiltered"
                />
              </template>
            </v-list-item>
            <v-divider v-if="filteredDocs.length > 0" />
            <v-list-item
              v-for="doc in filteredDocs"
              :key="doc.id"
              @click="toggle(doc.id)"
            >
              <template #prepend>
                <v-checkbox-btn
                  :model-value="draft.has(doc.id)"
                  @click.stop="toggle(doc.id)"
                />
              </template>
              <v-list-item-title :title="doc.filename" class="text-body-2">
                <v-icon size="small" class="mr-1">{{ mimeIcon(doc.mime_type) }}</v-icon>
                {{ doc.filename }}
              </v-list-item-title>
              <v-list-item-subtitle class="text-caption">
                {{ formatBytes(doc.size_bytes) }} · {{ formatDate(doc.uploaded_at) }}
              </v-list-item-subtitle>
            </v-list-item>
            <div v-if="filteredDocs.length === 0" class="pa-6 text-center text-body-2 text-medium-emphasis">
              {{ filter ? 'Ничего не найдено' : 'Нет проиндексированных документов' }}
            </div>
          </v-list>
        </div>
        <v-divider />
        <v-card-actions>
          <v-btn variant="text" :disabled="!isFiltered && draft.size === 0" @click="reset">
            Сбросить
          </v-btn>
          <v-spacer />
          <v-btn variant="text" @click="open = false">Отмена</v-btn>
          <v-btn color="primary" @click="apply">
            Применить{{ draft.size ? ` (${draft.size})` : '' }}
          </v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>
  </div>
</template>

<style scoped>
.scope-chip {
  cursor: pointer;
}
.scope-list {
  max-height: 50vh;
  overflow-y: auto;
}
</style>