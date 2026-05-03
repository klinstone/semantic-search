# SemSearch — фронтенд

Vue 3 + Vuetify 3 + Vite. Реализует UI для семантического поиска по документам:
поиск, загрузку, список документов с polling-статусом индексации и удаление.

## Стек

- **Vue 3** (Composition API, `<script setup>`)
- **Vuetify 3** (Material Design 3, MDI-иконки)
- **Vue Router 4** — роуты `/search` и `/documents`
- **Vite 6** — dev-сервер, прокси, сборка
- Без Pinia/axios: состояние в composables, HTTP — нативный `fetch` (+ `XMLHttpRequest` ради upload progress).

## Требования

- Node.js 20+
- Запущенный бэкенд на `http://localhost:8000` (см. корневой `docker-compose.yml`)

## Запуск в dev

```bash
cp .env.example .env
npm install
npm run dev
```

Открыть http://localhost:5173.
Запросы к `/api/*` Vite проксирует на `VITE_BACKEND_URL` (по умолчанию
`http://localhost:8000`). Базовый префикс API клиента — `/api/v1`.

## Production-сборка

```bash
npm run build       # собирает в dist/
npm run preview     # локально проверить готовый билд
```

## Docker

Двухстадийный `Dockerfile`: Node 20 для сборки → nginx 1.27 для отдачи.
Внутри контейнера nginx проксирует `/api/` на сервис `backend` в общей
docker-сети, поэтому в production-режиме никаких env-переменных не требуется.

Добавить в корневой `docker-compose.yml` (когда дойдёте до шага 11):

```yaml
services:
  frontend:
    build: ./frontend
    container_name: semsearch-frontend
    ports:
      - "127.0.0.1:5173:80"
    depends_on:
      - backend
    networks:
      - semsearch-network
    restart: unless-stopped
```

После этого UI будет на http://127.0.0.1:5173, а API — на
http://127.0.0.1:8000/api/v1 напрямую и через прокси на
http://127.0.0.1:5173/api/v1.

## Структура

```
frontend/
├── Dockerfile
├── nginx.conf
├── index.html
├── package.json
├── vite.config.js
├── .env.example
├── .gitignore
├── .dockerignore
└── src/
    ├── main.js                     # точка входа
    ├── App.vue                     # шапка + tabs + router-view + snackbar
    ├── router/index.js             # /search, /documents
    ├── plugins/vuetify.js          # тема, локаль, дефолты
    ├── styles/main.css             # глобальные мелочи
    ├── api/client.js               # fetch + ApiError + единый формат ошибок
    ├── utils/format.js             # bytes, dates, status labels/colors
    ├── composables/
    │   └── useNotification.js      # глобальный snackbar
    ├── views/
    │   ├── SearchView.vue          # POST /search + результаты
    │   └── DocumentsView.vue       # CRUD + polling статуса
    └── components/
        ├── DocumentUpload.vue      # drag-and-drop + progress
        └── SearchResultCard.vue    # карточка одного результата
```

## Соответствие API-контракту

| Эндпоинт                    | Где используется                                    |
| --------------------------- | --------------------------------------------------- |
| `GET /health`               | (доступен в `api.health()`, не вызывается в UI)     |
| `POST /documents`           | `DocumentUpload` → `api.uploadDocument`             |
| `GET /documents`            | `DocumentsView` → `api.listDocuments`               |
| `GET /documents/{id}`       | (доступен в `api.getDocument`, polling идёт по list)|
| `DELETE /documents/{id}`    | `DocumentsView` → `api.deleteDocument`              |
| `POST /search`              | `SearchView` → `api.search`                         |

Единый формат ошибок `{"error": {"code", "message", "details"}}` распаковывается
в `ApiError` (`src/api/client.js`); `message` показывается пользователю в
snackbar, `code` и `status` доступны программно, если понадобится разный UX
для конкретных кодов (например, 413 vs 415).

## Polling статуса индексации

На странице `/documents` каждые 2.5 секунды запрашивается список —
**только если** среди видимых элементов есть документы со статусом
`pending` или `processing`. Как только все документы переходят в
`indexed`/`failed`, polling простаивает (интервал не убивается, но
ничего не делает). Это соответствует пункту контракта про polling и
не требует WebSocket.

## Чего нет (и не должно быть в MVP)

- Аутентификации, ролей, мульти-юзерности
- Массовой загрузки
- WebSocket / SSE
- Тегов, коллекций, истории запросов
- Поиска по метаданным документа (фильтрация `document_ids` зарезервирована
  в API-контракте, но в UI пока не выставлена).
