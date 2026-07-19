<script>
  import { get } from 'svelte/store';
  import {
    galleryFilter,
    galleryState,
    filterToParams,
    patchGalleryItem,
    aspects,
    navigate,
  } from '../lib/stores.js';
  import { api } from '../lib/api.js';
  import { onGlobalKeys } from '../lib/keyboard.js';
  import FilterBar from '../components/FilterBar.svelte';
  import Thumb from '../components/Thumb.svelte';

  let loading = $state(false);
  let generation = 0;
  let lastKey = null;
  let sentinelEl = $state(null);

  const st = $derived($galleryState);

  // React to filter changes. On mount with an unchanged filter, restore the
  // previously loaded list (back-navigation) instead of refetching.
  $effect(() => {
    const key = JSON.stringify($galleryFilter);
    if (key === lastKey) return;
    lastKey = key;
    const cur = get(galleryState);
    if (cur.filterKey === key && cur.items.length) {
      const y = cur.scrollY;
      requestAnimationFrame(() => window.scrollTo(0, y));
      return;
    }
    resetAndLoad(key);
  });

  async function resetAndLoad(key) {
    const gen = ++generation;
    loading = true;
    galleryState.set({ items: [], nextCursor: null, selected: -1, filterKey: key, scrollY: 0 });
    window.scrollTo(0, 0);
    try {
      const p = filterToParams(get(galleryFilter));
      p.set('limit', '60');
      const res = await api.get(`/api/images?${p}`);
      if (gen !== generation) return;
      const items = res.items || [];
      galleryState.update((s) => ({
        ...s,
        items,
        nextCursor: res.nextCursor || null,
        selected: items.length ? 0 : -1,
      }));
    } catch {
      /* toasted by api */
    } finally {
      if (gen === generation) loading = false;
    }
  }

  async function loadMore() {
    const cur = get(galleryState);
    if (loading || !cur.nextCursor) return;
    const gen = ++generation;
    loading = true;
    try {
      const p = filterToParams(get(galleryFilter));
      p.set('limit', '60');
      p.set('cursor', cur.nextCursor);
      const res = await api.get(`/api/images?${p}`);
      if (gen !== generation) return;
      galleryState.update((s) => ({
        ...s,
        items: [...s.items, ...(res.items || [])],
        nextCursor: res.nextCursor || null,
      }));
    } catch {
      /* toasted by api */
    } finally {
      if (gen === generation) loading = false;
    }
  }

  // Infinite scroll sentinel.
  $effect(() => {
    if (!sentinelEl) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((en) => en.isIntersecting)) loadMore();
      },
      { rootMargin: '800px' }
    );
    io.observe(sentinelEl);
    return () => io.disconnect();
  });

  // Remember scroll position for back-restore.
  $effect(() => {
    return () => {
      galleryState.update((s) => ({ ...s, scrollY: window.scrollY }));
    };
  });

  function select(idx) {
    const cur = get(galleryState);
    if (!cur.items.length) return;
    const clamped = Math.max(0, Math.min(idx, cur.items.length - 1));
    galleryState.update((s) => ({ ...s, selected: clamped }));
    const el = document.querySelector(`[data-idx="${clamped}"]`);
    if (el) el.scrollIntoView({ block: 'nearest' });
    if (clamped >= cur.items.length - 8) loadMore();
  }

  function selectedItem() {
    const cur = get(galleryState);
    return cur.selected >= 0 ? cur.items[cur.selected] : null;
  }

  function open(item) {
    galleryState.update((s) => ({ ...s, scrollY: window.scrollY }));
    navigate(`/image/${item.id}`);
  }

  async function rate(item, target) {
    if (!item) return;
    const newScore = item.score === target ? 0 : target;
    const prev = item.score;
    patchGalleryItem(item.id, { score: newScore });
    try {
      // PUT rating carries the critique too — fetch it first so we never wipe it.
      let critique = '';
      if (item.hasCritique) {
        const d = await api.get(`/api/images/${item.id}`);
        critique = (d.rating && d.rating.critique) || '';
      }
      await api.put(`/api/images/${item.id}/rating`, { score: newScore, critique });
    } catch {
      patchGalleryItem(item.id, { score: prev });
    }
  }

  async function toggleAspectN(item, n) {
    if (!item) return;
    const list = get(aspects);
    const a = list[n - 1];
    if (!a) return;
    const prev = item.aspects || [];
    const has = prev.includes(a.name);
    const names = has ? prev.filter((x) => x !== a.name) : [...prev, a.name];
    const ids = names
      .map((name) => {
        const found = list.find((x) => x.name === name);
        return found ? found.id : null;
      })
      .filter((x) => x !== null);
    patchGalleryItem(item.id, { aspects: names });
    try {
      await api.put(`/api/images/${item.id}/aspects`, { aspectIds: ids });
    } catch {
      patchGalleryItem(item.id, { aspects: prev });
    }
  }

  $effect(() => {
    return onGlobalKeys((e) => {
      const cur = get(galleryState);
      switch (e.key) {
        case 'j':
        case 'ArrowRight':
          select(cur.selected + 1);
          return true;
        case 'k':
        case 'ArrowLeft':
          select(cur.selected - 1);
          return true;
        case 'l':
          rate(selectedItem(), 1);
          return true;
        case 'x':
          rate(selectedItem(), -1);
          return true;
        case 'Enter': {
          const it = selectedItem();
          if (it) open(it);
          return true;
        }
        case 'u':
          galleryFilter.update((f) => ({ ...f, score: f.score === 'unrated' ? 'all' : 'unrated' }));
          return true;
        default:
          if (/^[1-9]$/.test(e.key)) {
            toggleAspectN(selectedItem(), Number(e.key));
            return true;
          }
      }
    });
  });
</script>

<FilterBar />

<main class="gallery">
  {#if st.items.length}
    <div class="grid">
      {#each st.items as item, i (item.id)}
        <Thumb {item} index={i} selected={i === st.selected} onselect={select} onopen={open} />
      {/each}
    </div>
  {:else if !loading}
    <p class="empty">No images match the current filter.</p>
  {/if}

  <div class="sentinel" bind:this={sentinelEl}>
    {#if loading}
      <span class="dim">loading…</span>
    {:else if st.items.length && !st.nextCursor}
      <span class="dim">— end · {st.items.length} images —</span>
    {/if}
  </div>
</main>
