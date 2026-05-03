<script setup>
import { useNotification } from '@/composables/useNotification'

const { state: snack } = useNotification()
</script>

<template>
  <v-app>
    <v-app-bar color="surface" flat border density="comfortable">
      <v-container class="d-flex align-center pa-0" fluid style="max-width: 1200px">
        <router-link to="/search" class="brand text-decoration-none d-flex align-center">
          <v-icon color="primary" class="mr-2">mdi-text-search-variant</v-icon>
          <span class="text-h6 font-weight-bold text-primary">SemSearch</span>
          <span class="text-body-2 text-medium-emphasis ml-3 d-none d-md-inline">
            семантический поиск по документам
          </span>
        </router-link>
        <v-spacer />
        <v-tabs density="comfortable" hide-slider color="primary">
          <v-tab to="/search" prepend-icon="mdi-magnify">Поиск</v-tab>
          <v-tab to="/documents" prepend-icon="mdi-file-document-multiple-outline">
            Документы
          </v-tab>
        </v-tabs>
      </v-container>
    </v-app-bar>

    <v-main class="bg-background">
      <v-container style="max-width: 1200px">
        <router-view v-slot="{ Component }">
          <transition name="fade" mode="out-in">
            <keep-alive include="SearchView">
              <component :is="Component" />
            </keep-alive>
          </transition>
        </router-view>
      </v-container>
    </v-main>

    <v-snackbar
      v-model="snack.visible"
      :color="snack.color"
      :timeout="snack.timeout"
      location="bottom right"
    >
      {{ snack.message }}
      <template #actions>
        <v-btn variant="text" @click="snack.visible = false">Закрыть</v-btn>
      </template>
    </v-snackbar>
  </v-app>
</template>

<style scoped>
.brand {
  color: inherit;
}
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.18s ease;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
