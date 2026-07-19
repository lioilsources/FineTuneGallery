// Single same-origin fetch wrapper: JSON in/out, error toast on any failure.
import { toast } from './stores.js';

async function request(method, path, body) {
  const init = { method };
  if (body !== undefined) {
    init.headers = { 'Content-Type': 'application/json' };
    init.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(path, init);
  } catch (err) {
    toast.error(`Network error — ${err.message}`);
    throw err;
  }
  if (!res.ok) {
    let detail = '';
    try {
      const text = await res.text();
      if (text) {
        try {
          detail = JSON.parse(text).error || text;
        } catch {
          detail = text;
        }
      }
    } catch {
      /* body unreadable — status is enough */
    }
    const msg = `${method} ${path} → ${res.status}${detail ? ` — ${String(detail).slice(0, 180)}` : ''}`;
    toast.error(msg);
    throw new Error(msg);
  }
  if (res.status === 204) return null;
  const ct = res.headers.get('content-type') || '';
  return ct.includes('json') ? res.json() : res.text();
}

export const api = {
  get: (path) => request('GET', path),
  post: (path, body) => request('POST', path, body),
  put: (path, body) => request('PUT', path, body),
  del: (path) => request('DELETE', path),
};
