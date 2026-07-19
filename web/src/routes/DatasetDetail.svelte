<script>
  import { get } from 'svelte/store';
  import { meta, aspects, ensureMeta, ensureAspects, galleryFilter, filterToParams, toast } from '../lib/stores.js';
  import { api } from '../lib/api.js';

  let { id } = $props();

  ensureMeta();
  ensureAspects();

  let data = $state(null); // { dataset, items }
  let error = $state(false);
  let building = $state(false);
  let adding = $state(false);
  let builtInfo = $state(null); // { path, imageCount } from the last build in this session
  let generation = 0;

  $effect(() => {
    load(id);
  });

  async function load(dsId) {
    const gen = ++generation;
    error = false;
    try {
      const d = await api.get(`/api/datasets/${dsId}`);
      if (gen === generation) data = d;
    } catch {
      if (gen === generation) error = true;
    }
  }

  const ds = $derived(data ? data.dataset : null);
  const items = $derived((data && data.items) || []);

  const aspectName = $derived.by(() => {
    if (!ds) return '';
    const a = $aspects.find((x) => x.id === ds.aspectId);
    return a ? a.name : String(ds.aspectId ?? '');
  });

  const baseModelLabel = $derived.by(() => {
    if (!ds) return '';
    const m = ($meta.models || []).find((x) => x.id === ds.baseModel);
    return m ? m.label : ds.baseModel;
  });

  const isBuilt = $derived(!!ds && (ds.status === 'built' || !!ds.builtAt));

  async function build() {
    building = true;
    try {
      const res = await api.post(`/api/datasets/${id}/build`);
      builtInfo = res;
      toast.info(`Built ${res.imageCount} images → ${res.path}`);
      await load(id);
    } catch {
      /* toasted by api */
    } finally {
      building = false;
    }
  }

  // Adds whatever the gallery filter currently selects — uses the shared
  // filter store, so set the filter in the gallery first, then click here.
  async function addFromFilter() {
    adding = true;
    try {
      const qs = filterToParams(get(galleryFilter)).toString();
      const res = await api.post(`/api/datasets/${id}/items/from-filter${qs ? `?${qs}` : ''}`);
      toast.info(`Added ${res.added} image${res.added === 1 ? '' : 's'} from the current gallery filter`);
      await load(id);
    } catch {
      /* toasted by api */
    } finally {
      adding = false;
    }
  }

  async function removeItem(it) {
    try {
      await api.post(`/api/datasets/${id}/items`, { add: [], remove: [it.imageId] });
      data.items = data.items.filter((x) => x.imageId !== it.imageId);
      if (data.dataset && typeof data.dataset.itemCount === 'number') {
        data.dataset.itemCount = Math.max(0, data.dataset.itemCount - 1);
      }
    } catch {
      /* toasted by api */
    }
  }

  async function saveField(patch) {
    try {
      await api.put(`/api/datasets/${id}`, patch);
      data.dataset = { ...data.dataset, ...patch };
      toast.info('Saved');
    } catch {
      /* toasted by api */
    }
  }
</script>

<main class="page">
  {#if error}
    <p class="empty">Failed to load dataset {id}. <a href="#/datasets">back to datasets</a></p>
  {:else if !ds}
    <p class="dim">loading…</p>
  {:else}
    <div class="ds-head">
      <div>
        <h2>{ds.name}</h2>
        <div class="ds-meta dim">
          aspect <b>{aspectName}</b> · base <b>{baseModelLabel}</b> ·
          <span class="status s-{ds.status}">{ds.status}</span>
          · {ds.itemCount ?? items.length} items
          {#if ds.builtAt}· built {new Date(ds.builtAt).toLocaleString()}{/if}
        </div>
        <div class="ds-edit">
          <label class="dim"
            >trigger
            <input
              class="inp small"
              value={ds.triggerWord}
              onchange={(e) => saveField({ triggerWord: e.target.value })}
            /></label
          >
          <label class="dim"
            >repeats
            <input
              class="inp num small"
              type="number"
              min="1"
              value={ds.repeats}
              onchange={(e) => saveField({ repeats: Number(e.target.value) || ds.repeats })}
            /></label
          >
        </div>
      </div>
      <div class="ds-actions">
        <button class="btn" onclick={addFromFilter} disabled={adding}>
          {adding ? 'adding…' : 'Add current gallery filter'}
        </button>
        <button class="btn" onclick={build} disabled={building}>{building ? 'building…' : 'Build'}</button>
        {#if isBuilt}
          <a class="btn" href={`/api/datasets/${id}/download`}>Download</a>
        {/if}
      </div>
    </div>

    {#if builtInfo}
      <p class="hint">Built: <code>{builtInfo.path}</code> ({builtInfo.imageCount} images)</p>
    {/if}

    <h3 style="margin-top: 16px">{items.length} items</h3>
    {#if !items.length}
      <p class="empty">No items yet — set a filter in the gallery, then use “Add current gallery filter”.</p>
    {:else}
      <div class="grid small" style="padding: 0">
        {#each items as it (it.imageId)}
          <div class="dcell">
            <a href={`#/image/${it.imageId}`}>
              <img
                class="dthumb"
                src="/thumb/{it.sha256}"
                alt={it.caption || `image ${it.imageId}`}
                loading="lazy"
                onerror={(e) => e.currentTarget.classList.add('broken')}
              />
            </a>
            {#if it.score === 1}<span class="badge b-like corner">♥</span>{/if}
            {#if it.score === -1}<span class="badge b-dislike corner">✕</span>{/if}
            <button class="cell-x" title="remove from dataset" onclick={() => removeItem(it)}>✕</button>
            <div class="cap" title={it.caption || ''}>{it.caption || '(no caption)'}</div>
          </div>
        {/each}
      </div>
    {/if}
  {/if}
</main>
