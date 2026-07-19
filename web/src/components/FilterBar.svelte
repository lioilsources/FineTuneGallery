<script>
  import {
    galleryFilter,
    emptyFilter,
    meta,
    aspects,
    sessions,
    galleryState,
    ensureMeta,
    ensureAspects,
    ensureSessions,
    toast,
  } from '../lib/stores.js';
  import { api } from '../lib/api.js';

  ensureMeta();
  ensureAspects();
  ensureSessions();

  const f = $derived($galleryFilter);

  function setF(patch) {
    galleryFilter.update((cur) => ({ ...cur, ...patch }));
  }

  function toggleAspect(name) {
    setF({ aspect: f.aspect === name ? '' : name });
  }

  function clearAll() {
    galleryFilter.set({ ...emptyFilter });
  }

  const anyActive = $derived(
    f.model !== '' || f.aspect !== '' || f.score !== 'all' || f.session !== '' || f.uncaptioned || f.criterion !== ''
  );

  const summary = $derived.by(() => {
    const parts = [];
    if (f.model) {
      const m = ($meta.models || []).find((x) => x.id === f.model);
      parts.push(m ? m.label : f.model);
    }
    if (f.aspect) parts.push(`#${f.aspect}`);
    if (f.score === 'liked') parts.push('♥ liked');
    if (f.score === 'disliked') parts.push('✕ disliked');
    if (f.score === 'unrated') parts.push('unrated');
    if (f.session) {
      const s = $sessions.find((x) => String(x.id) === String(f.session));
      parts.push(`session: ${s ? s.title : f.session}`);
    }
    if (f.uncaptioned) parts.push('uncaptioned');
    if (f.criterion) parts.push(f.criterion.replace(':1', ' +').replace(':-1', ' −'));
    return parts.join(' · ');
  });

  // inline "new aspect" mini-form
  let addingAspect = $state(false);
  let newAspectName = $state('');
  let newAspectTrigger = $state('');

  async function createAspect() {
    const name = newAspectName.trim();
    if (!name) return;
    try {
      await api.post('/api/aspects', { name, triggerWord: newAspectTrigger.trim() });
      toast.info(`Aspect "${name}" created`);
      newAspectName = '';
      newAspectTrigger = '';
      addingAspect = false;
      await ensureAspects(true);
    } catch {
      /* toasted by api */
    }
  }

  function aspectFormKeys(e) {
    if (e.key === 'Enter') createAspect();
    if (e.key === 'Escape') addingAspect = false;
    e.stopPropagation();
  }

  const scoreOpts = [
    ['all', 'All'],
    ['liked', '♥ Liked'],
    ['disliked', '✕ Disliked'],
    ['unrated', 'Unrated'],
  ];
</script>

<div class="filterbar">
  <div class="filter-row">
    <select class="sel" value={f.model} onchange={(e) => setF({ model: e.target.value })} title="Model">
      <option value="">All models</option>
      {#each $meta.models as mm (mm.id)}
        <option value={mm.id}>{mm.label}</option>
      {/each}
    </select>

    <div class="seg" role="group" aria-label="Score filter">
      {#each scoreOpts as [val, label] (val)}
        <button
          class="seg-btn"
          class:active={f.score === val}
          onclick={(e) => {
            setF({ score: val });
            e.currentTarget.blur();
          }}>{label}</button
        >
      {/each}
    </div>

    <select class="sel" value={f.session} onchange={(e) => setF({ session: e.target.value })} title="Session">
      <option value="">All sessions</option>
      {#each $sessions as s (s.id)}
        <option value={s.id}>{s.title} ({s.imageCount})</option>
      {/each}
    </select>

    <select class="sel" value={f.criterion} onchange={(e) => setF({ criterion: e.target.value })} title="Criterion filter">
      <option value="">Any criterion</option>
      {#each $meta.criteria as c (c)}
        <option value="{c}:1">{c} +</option>
        <option value="{c}:-1">{c} −</option>
      {/each}
    </select>

    <button
      class="chip"
      class:active={f.uncaptioned}
      title="Only images without a caption"
      onclick={(e) => {
        setF({ uncaptioned: !f.uncaptioned });
        e.currentTarget.blur();
      }}>uncaptioned</button
    >

    {#if anyActive}
      <button class="chip ghost" onclick={clearAll}>clear</button>
    {/if}
  </div>

  <div class="filter-row">
    <span class="lbl">aspects</span>
    {#each $aspects as a, i (a.id)}
      <button
        class="chip"
        class:active={f.aspect === a.name}
        title={i < 9 ? `toggle on selected image: key ${i + 1}` : a.name}
        onclick={(e) => {
          toggleAspect(a.name);
          e.currentTarget.blur();
        }}>{a.name}</button
      >
    {/each}
    {#if addingAspect}
      <input class="inp small" placeholder="name" bind:value={newAspectName} onkeydown={aspectFormKeys} />
      <input class="inp small" placeholder="trigger word" bind:value={newAspectTrigger} onkeydown={aspectFormKeys} />
      <button class="chip" onclick={createAspect}>add</button>
      <button class="chip ghost" onclick={() => (addingAspect = false)}>cancel</button>
    {:else}
      <button class="chip ghost" title="New aspect" onclick={() => (addingAspect = true)}>+</button>
    {/if}
  </div>

  <div class="filter-row summary-row">
    <span>{$galleryState.items.length} loaded{$galleryState.nextCursor ? '+' : ''}</span>
    {#if summary}
      <span>·</span>
      <span class="summary">{summary}</span>
    {/if}
    <span class="spacer"></span>
    <a class="chip mini ghost" href="#/datasets">datasets</a>
    <a class="chip mini ghost" href="#/stats">stats</a>
  </div>
</div>
