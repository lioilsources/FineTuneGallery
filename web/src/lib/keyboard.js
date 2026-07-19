// Global keyboard helper — window keydown, skipping events originating in editable targets.

export function isEditable(el) {
  if (!el) return false;
  const tag = el.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable;
}

// handler(e) should return true when it handled the key (=> preventDefault).
// Returns an unsubscribe function — use as an $effect cleanup.
export function onGlobalKeys(handler) {
  const listener = (e) => {
    if (isEditable(e.target)) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (handler(e) === true) e.preventDefault();
  };
  window.addEventListener('keydown', listener);
  return () => window.removeEventListener('keydown', listener);
}
