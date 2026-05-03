import { createRouter, createWebHistory } from 'vue-router'

import SearchView from '@/views/SearchView.vue'
import DocumentsView from '@/views/DocumentsView.vue'

const routes = [
  { path: '/', redirect: '/search' },
  {
    path: '/search',
    name: 'search',
    component: SearchView,
    meta: { title: 'Поиск' }
  },
  {
    path: '/documents',
    name: 'documents',
    component: DocumentsView,
    meta: { title: 'Документы' }
  },
  { path: '/:pathMatch(.*)*', redirect: '/search' }
]

const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior() {
    return { top: 0 }
  }
})

router.afterEach((to) => {
  if (to.meta?.title) {
    document.title = `${to.meta.title} — SemSearch`
  }
})

export default router
