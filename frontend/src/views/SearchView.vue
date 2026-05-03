<script setup>
defineOptions({ name: 'SearchView' })
import { ref, computed } from 'vue'
import { api } from '@/api/client'
import { useNotification } from '@/composables/useNotification'
import SearchResultCard from '@/components/SearchResultCard.vue'

const notify = useNotification()

const query = ref('')
const limit = ref(10)
const loading = ref(false)
const response = ref(null)     // last full response from POST /search
const lastQuery = ref('')

const hasResults = computed(() => !!response.value && response.value.results.length > 0)
const isEmpty = computed(() => !!response.value && response.value.results.length === 0)
const isInitial = computed(() => !loading.value && response.value === null)

async function runSearch() {
  const q = query.value.trim()
  if (!q) return
  if (q.length > 1000) {
    notify.warning('Запрос не должен превышать 1000 символов')
    return
  }

  loading.value = true
  try {
    response.value = await api.search(q, { limit: limit.value })
    lastQuery.value = q
  } catch (e) {
    notify.error(e.message || 'Ошибка поиска')
    response.value = null
  } finally {
    loading.value = false
  }
}

function clearQuery() {
  query.value = ''
  response.value = null
  lastQuery.value = ''
}
</script>

<template>
  <div>
    <div class="text-center mb-6 mt-4">
      <h1 class="text-h4 font-weight-bold mb-2">Семантический поиск</h1>
      <p class="text-body-1 text-medium-emphasis mb-0">
        Поиск релевантных фрагментов по смыслу запроса, а не по точному совпадению слов
      </p>
    </div>

    <v-card class="mb-6 pa-4" elevation="2">
      <v-form @submit.prevent="runSearch">
        <div class="search-row">
          <v-text-field
            v-model="query"
            placeholder="Например: алгоритмы машинного обучения для классификации текстов"
            prepend-inner-icon="mdi-magnify"
            counter="1000"
            maxlength="1000"
            hide-details="auto"
            autofocus
            clearable
            class="flex-grow-1"
            @click:clear="clearQuery"
          />
          <v-select
            v-model="limit"
            :items="[5, 10, 20, 30, 50]"
            label="Топ"
            hide-details
            style="max-width: 110px"
          />
          <v-btn
            color="primary"
            size="large"
            type="submit"
            :loading="loading"
            :disabled="!query.trim()"
            prepend-icon="mdi-magnify"
          >
            Найти
          </v-btn>
        </div>
      </v-form>
    </v-card>

    <div v-if="response" class="d-flex align-center justify-space-between mb-3 px-1">
      <div class="text-body-2 text-medium-emphasis">
        Найдено фрагментов: <strong class="text-primary">{{ response.total_found }}</strong>
        <template v-if="response.took_ms != null">
          · за {{ response.took_ms }} мс
        </template>
      </div>
      <div v-if="lastQuery" class="text-body-2 text-medium-emphasis text-truncate" style="max-width: 50%">
        запрос: «{{ lastQuery }}»
      </div>
    </div>

    <v-progress-linear v-if="loading" indeterminate color="primary" class="mb-4" />

    <div v-if="hasResults" class="results-list">
      <SearchResultCard
        v-for="(item, idx) in response.results"
        :key="item.chunk_id"
        :item="item"
        :rank="idx + 1"
      />
    </div>

    <v-card v-else-if="isEmpty" class="pa-8 text-center" variant="outlined">
      <v-icon size="48" color="grey">mdi-text-search</v-icon>
      <div class="text-h6 mt-3">Ничего не найдено</div>
      <div class="text-body-2 text-medium-emphasis mt-1">
        Попробуйте переформулировать запрос или загрузите больше документов
      </div>
      <v-btn class="mt-4" variant="text" to="/documents" prepend-icon="mdi-upload">
        Перейти к документам
      </v-btn>
    </v-card>

    <v-card v-else-if="isInitial" class="pa-10 text-center" variant="flat" color="transparent">
      <v-icon size="64" color="grey-lighten-1">mdi-text-search-variant</v-icon>
      <div class="text-body-1 text-medium-emphasis mt-3">
        Введите запрос — например, описание темы или вопрос
      </div>
    </v-card>
  </div>
</template>

<style scoped>
.search-row {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  flex-wrap: wrap;
}
.search-row > .v-text-field {
  min-width: 240px;
}
.results-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
</style>
