<script setup>
import { computed } from 'vue'

const props = defineProps({
  item: { type: Object, required: true },
  rank: { type: Number, required: true }
})

const scorePercent = computed(() => Math.round((props.item.score ?? 0) * 100))

const scoreColor = computed(() => {
  const s = props.item.score ?? 0
  if (s >= 0.75) return 'success'
  if (s >= 0.5) return 'info'
  if (s >= 0.3) return 'warning'
  return 'grey'
})

const pageInfo = computed(() => {
  const md = props.item.metadata
  if (!md) return null
  if (md.page != null) return `с. ${md.page}`
  return null
})
</script>

<template>
  <v-card variant="outlined" class="pa-4 result-card">
    <div class="result-head">
      <div class="rank-badge">#{{ rank }}</div>
      <v-icon size="small" color="primary">mdi-file-document-outline</v-icon>
      <div class="filename text-body-2 font-weight-medium" :title="item.document_filename">
        {{ item.document_filename }}
      </div>
      <v-chip size="x-small" variant="tonal" color="grey">
        фрагмент {{ item.chunk_index }}
      </v-chip>
      <v-chip v-if="pageInfo" size="x-small" variant="tonal" color="grey">
        {{ pageInfo }}
      </v-chip>
      <v-spacer />
      <v-chip :color="scoreColor" size="small" variant="flat">
        <v-icon start size="small">mdi-target</v-icon>
        {{ scorePercent }}%
      </v-chip>
    </div>

    <div class="result-text text-body-1">{{ item.text }}</div>
  </v-card>
</template>

<style scoped>
.result-card {
  transition: border-color 0.15s ease;
}
.result-card:hover {
  border-color: rgb(var(--v-theme-primary));
}
.result-head {
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
  margin-bottom: 10px;
}
.rank-badge {
  font-family: ui-monospace, 'SF Mono', Menlo, Consolas, monospace;
  font-weight: 700;
  color: rgba(0, 0, 0, 0.45);
  font-size: 13px;
  min-width: 28px;
}
.filename {
  max-width: 60%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.result-text {
  white-space: pre-wrap;
  line-height: 1.55;
  color: rgba(0, 0, 0, 0.82);
}
</style>
