import { writable } from 'svelte/store';
import { api } from './api.js';

/* ---------------- toasts ---------------- */

export const toasts = writable([]);
let toastSeq = 0;

function pushToast(msg, kind, ttl) {
  const id = ++toastSeq;
  toasts.update((list) => [...list, { id, msg, kind }]);
  setTimeout(() => toasts.update((list) => list.filter((t) => t.id !== id)), ttl);
}

export const toast = {
  info: (msg) => pushToast(msg, 'info', 3500),
  error: (msg) => pushToast(msg, 'error', 6500),
};

/* ---------------- hash router ---------------- */

function currentPath() {
  const raw = window.location.hash.replace(/^#/, '');
  return raw.startsWith('/') ? raw : `/${raw}`;
}

export const route = writable(currentPath());
window.addEventListener('hashchange', () => route.set(currentPath()));

export function navigate(path) {
  if (currentPath() === path) return;
  window.location.hash = path;
}

export function matchRoute(path) {
  if (path === '/' || path === '') return { name: 'gallery' };
  let m;
  if ((m = path.match(/^\/image\/([^/]+)$/))) return { name: 'image', id: decodeURIComponent(m[1]) };
  if (path === '/datasets') return { name: 'datasets' };
  if ((m = path.match(/^\/datasets\/([^/]+)$/))) return { name: 'dataset', id: decodeURIComponent(m[1]) };
  if (path === '/stats') return { name: 'stats' };
  return { name: 'gallery' };
}

/* ---------------- meta / aspects / sessions caches ---------------- */

export const meta = writable({ models: [], criteria: [] });
export const aspects = writable([]);
export const sessions = writable([]);

let metaP = null;
export function ensureMeta() {
  if (!metaP) {
    metaP = api
      .get('/api/meta')
      .then((v) => meta.set(v))
      .catch(() => {
        metaP = null;
      });
  }
  return metaP;
}

let aspectsP = null;
export function ensureAspects(force = false) {
  if (!aspectsP || force) {
    aspectsP = api
      .get('/api/aspects')
      .then((v) => aspects.set(v))
      .catch(() => {
        aspectsP = null;
      });
  }
  return aspectsP;
}

let sessionsP = null;
export function ensureSessions(force = false) {
  if (!sessionsP || force) {
    sessionsP = api
      .get('/api/sessions')
      .then((v) => sessions.set(v))
      .catch(() => {
        sessionsP = null;
      });
  }
  return sessionsP;
}

/* ---------------- gallery filter (shared — also drives dataset "from-filter") ---------------- */

export const emptyFilter = Object.freeze({
  model: '',
  aspect: '', // aspect name
  score: 'all', // 'all' | 'liked' | 'disliked' | 'unrated'
  session: '',
  uncaptioned: false,
  criterion: '', // '' | '<name>:1' | '<name>:-1'
});

export const galleryFilter = writable({ ...emptyFilter });

export function filterToParams(f) {
  const p = new URLSearchParams();
  if (f.model) p.set('model', f.model);
  if (f.aspect) p.set('aspect', f.aspect);
  if (f.score === 'liked') p.set('score', '1');
  else if (f.score === 'disliked') p.set('score', '-1');
  else if (f.score === 'unrated') p.set('unrated', '1');
  if (f.session) p.set('session', String(f.session));
  if (f.uncaptioned) p.set('captioned', '0');
  if (f.criterion) p.set('criterion', f.criterion);
  return p;
}

/* ---------------- gallery result list ----------------
   Survives navigation: back-restore of the grid + prev/next in Detail. */

export const galleryState = writable({
  items: [],
  nextCursor: null,
  selected: -1,
  filterKey: '', // JSON of the filter the list was loaded with ('' = never loaded)
  scrollY: 0,
});

export function patchGalleryItem(id, patch) {
  galleryState.update((st) => ({
    ...st,
    items: st.items.map((it) => (String(it.id) === String(id) ? { ...it, ...patch } : it)),
  }));
}
