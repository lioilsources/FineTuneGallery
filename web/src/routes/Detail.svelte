<script>
  import { get } from 'svelte/store';
  import {
    meta,
    aspects,
    ensureMeta,
    ensureAspects,
    galleryState,
    galleryFilter,
    navigate,
    toast,
    patchGalleryItem,
  } from '../lib/stores.js';
  import { api } from '../lib/api.js';
  import { onGlobalKeys } from '../lib/keyboard.js';

  let { id } = $props();

  ensureMeta();
  ensureAspects();

  let data = $state(null);
  let error = $state(false);
  let critiqueText = $state('');
  let captionText = $state('');
  let critiqueSaved = $state(false);
  let captionSaved = $state(false);
  let autocaptionPending = $state(false);
  let generation = 0;

  $effect(() => {
    load(id);
  });

  async function load(imageId) {
    const gen = ++generation;
    error = false;
    data = null;
    window.scrollTo(0, 0);
    try {
      const d = await api.get(`/api/images/${imageId}`);
      if (gen !== generation) return;
      data = d;
      critiqueText = (d.rating && d.rating.critique) || '';
      captionText = (d.captions && d.captions.human && d.captions.human.text) || '';
    } catch {
      if (gen === generation) error = true;
    }
  }

  const img = $derived(data ? data.image : null);
  const node = $derived(data ? data.node : null);
  const chain = $derived((data && data.parentChain) || []);

  const visibleCriteria = $derived.by(() => {
    if (!data) return [];
    return ($meta.criteria || []).filter((c) => {
      if (c === 'pose_adherence') return !!(node && node.poseId);
      if (c.startsWith('source_')) return chain.length > 0;
      return true;
    });
  });

  const modelLabel = $derived.by(() => {
    if (!node) return '—';
    const m = ($meta.models || []).find((x) => x.id === node.modelId);
    return m ? m.label : node.modelId || '—';
  });

  /* ---------------- rating ---------------- */

  async function rate(target) {
    if (!data) return;
    const curScore = (data.rating && data.rating.score) || 0;
    const newScore = curScore === target ? 0 : target;
    data.rating = { ...(data.rating || {}), score: newScore };
    patchGalleryItem(id, { score: newScore });
    try {
      await api.put(`/api/images/${id}/rating`, { score: newScore, critique: critiqueText });
    } catch {
      data.rating = { ...(data.rating || {}), score: curScore };
      patchGalleryItem(id, { score: curScore });
    }
  }

  /* ---------------- criteria ---------------- */

  async function setCriterion(name, target) {
    if (!data) return;
    const prev = { ...(data.criteria || {}) };
    const cur = prev[name];
    const newScore = cur === target ? 0 : target;
    const next = { ...prev };
    if (newScore === 0) delete next[name];
    else next[name] = newScore;
    data.criteria = next;
    try {
      await api.put(`/api/images/${id}/criteria`, { criterion: name, score: newScore });
    } catch {
      data.criteria = prev;
    }
  }

  /* ---------------- aspects ---------------- */

  async function toggleAspect(aspectId) {
    if (!data) return;
    const cur = data.aspects || [];
    const has = cur.includes(aspectId);
    const next = has ? cur.filter((x) => x !== aspectId) : [...cur, aspectId];
    data.aspects = next;
    const names = next
      .map((aid) => {
        const found = get(aspects).find((a) => a.id === aid);
        return found ? found.name : null;
      })
      .filter(Boolean);
    patchGalleryItem(id, { aspects: names });
    try {
      await api.put(`/api/images/${id}/aspects`, { aspectIds: next });
    } catch {
      data.aspects = cur;
    }
  }

  /* ---------------- debounced saves (800 ms) ---------------- */

  function autogrow(e) {
    const t = e.target;
    t.style.height = 'auto';
    t.style.height = Math.min(t.scrollHeight + 2, 320) + 'px';
  }

  let critiqueTimer = null;
  function critiqueChanged(e) {
    if (e) autogrow(e);
    if (critiqueTimer) clearTimeout(critiqueTimer);
    const imageId = id; // capture — survives prev/next navigation
    critiqueTimer = setTimeout(async () => {
      critiqueTimer = null;
      const score = (data && data.rating && data.rating.score) || 0;
      try {
        await api.put(`/api/images/${imageId}/rating`, { score, critique: critiqueText });
        patchGalleryItem(imageId, { hasCritique: critiqueText.trim() !== '' });
        critiqueSaved = true;
        setTimeout(() => (critiqueSaved = false), 1600);
      } catch {
        /* toasted by api */
      }
    }, 800);
  }

  let captionTimer = null;
  function captionChanged(e) {
    if (e) autogrow(e);
    if (captionTimer) clearTimeout(captionTimer);
    const imageId = id;
    captionTimer = setTimeout(async () => {
      captionTimer = null;
      try {
        await api.put(`/api/images/${imageId}/caption`, { text: captionText });
        if (data && data.captions) {
          data.captions.human = captionText === '' ? null : { text: captionText };
        }
        patchGalleryItem(imageId, {
          caption: {
            auto: !!(data && data.captions && data.captions.auto),
            human: captionText !== '',
            refined: !!(data && data.captions && data.captions.refined),
          },
        });
        captionSaved = true;
        setTimeout(() => (captionSaved = false), 1600);
      } catch {
        /* toasted by api */
      }
    }, 800);
  }

  /* ---------------- captions ---------------- */

  const autoTags = $derived.by(() => {
    const auto = data && data.captions && data.captions.auto;
    if (!auto || !auto.text) return [];
    return auto.text
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean);
  });

  function appendTag(tag) {
    const cur = captionText.trim();
    captionText = cur === '' ? tag : `${cur.replace(/,\s*$/, '')}, ${tag}`;
    captionChanged();
  }

  async function autocaption() {
    try {
      await api.post(`/api/images/${id}/autocaption`);
      autocaptionPending = true;
      toast.info('Auto-caption queued — refreshing in 3 s…');
      const imageId = id;
      setTimeout(async () => {
        try {
          const d = await api.get(`/api/images/${imageId}`);
          if (String(imageId) === String(id) && data) {
            data.captions.auto = (d.captions && d.captions.auto) || null;
            data.captions.refined = (d.captions && d.captions.refined) || null;
          }
        } catch {
          /* toasted by api */
        } finally {
          if (String(imageId) === String(id)) autocaptionPending = false;
        }
      }, 3000);
    } catch {
      /* toasted by api */
    }
  }

  /* ---------------- navigation ---------------- */

  const listPos = $derived.by(() => {
    const s = $galleryState;
    const idx = s.items.findIndex((it) => String(it.id) === String(id));
    return { idx, count: s.items.length };
  });

  function go(delta) {
    const s = get(galleryState);
    const idx = s.items.findIndex((it) => String(it.id) === String(id));
    if (idx < 0) return;
    const next = idx + delta;
    if (next < 0 || next >= s.items.length) return;
    galleryState.update((x) => ({ ...x, selected: next }));
    navigate(`/image/${s.items[next].id}`);
  }

  function goBack() {
    if (window.history.length > 1) window.history.back();
    else navigate('/');
  }

  function openSession() {
    if (!img || img.sessionId == null) return;
    galleryFilter.update((f) => ({ ...f, session: String(img.sessionId) }));
    navigate('/');
  }

  function copyPrompt() {
    if (!node || !node.prompt) return;
    navigator.clipboard.writeText(node.prompt).then(
      () => toast.info('Prompt copied'),
      () => toast.error('Copy failed')
    );
  }

  $effect(() => {
    return onGlobalKeys((e) => {
      switch (e.key) {
        case 'l':
          rate(1);
          return true;
        case 'x':
          rate(-1);
          return true;
        case 'Escape':
          goBack();
          return true;
        case 'j':
        case 'ArrowRight':
          go(1);
          return true;
        case 'k':
        case 'ArrowLeft':
          go(-1);
          return true;
      }
    });
  });

  function fmtDate(s) {
    if (!s) return '—';
    const d = new Date(s);
    return isNaN(d.getTime()) ? s : d.toLocaleString();
  }
</script>

{#if error}
  <main class="detail">
    <p class="empty">
      Failed to load image {id}.
      <button class="chip" onclick={() => load(id)}>retry</button>
      <a class="chip ghost" href="#/">gallery</a>
    </p>
  </main>
{:else if !data}
  <main class="detail"><p class="empty">loading…</p></main>
{:else}
  <main class="detail">
    <div class="detail-left">
      <img class="hero" src="/img/{img.sha256}" alt={node.prompt || `image ${img.id}`} />
      <div class="detail-nav">
        <button class="chip" onclick={() => go(-1)} disabled={listPos.idx <= 0}>‹ prev</button>
        <span class="dim">{listPos.idx >= 0 ? `${listPos.idx + 1} / ${listPos.count}` : ''}</span>
        <button class="chip" onclick={() => go(1)} disabled={listPos.idx < 0 || listPos.idx >= listPos.count - 1}
          >next ›</button
        >
        <button class="chip ghost" onclick={goBack}>back <kbd>esc</kbd></button>
      </div>
    </div>

    <div class="detail-right">
      {#if chain.length}
        <section>
          <h3>Source</h3>
          <a class="source-row" href={`#/image/${chain[0].imageId}`}>
            <img class="source-thumb" src="/thumb/{chain[0].sha256}" alt="img2img source" />
            <span class="source-prompt dim">{chain[0].prompt || '(no prompt)'}</span>
          </a>
          {#if chain.length > 1}
            <div class="chain-strip">
              {#each chain as link, i (link.imageId)}
                <a href={`#/image/${link.imageId}`} title={link.prompt || `ancestor ${i + 1}`}>
                  <img src="/thumb/{link.sha256}" alt={`ancestor ${i + 1}`} />
                </a>
              {/each}
            </div>
          {/if}
        </section>
      {/if}

      {#if node.poseId}
        <section>
          <h3>Pose</h3>
          <img class="pose-thumb" src="/poses/{node.poseId}.png" alt={`pose ${node.poseId}`} />
        </section>
      {/if}

      <section>
        <div class="rate-row">
          <button class="rate-btn like" class:active={data.rating && data.rating.score === 1} onclick={() => rate(1)}>
            ♥ Like <kbd>l</kbd>
          </button>
          <button
            class="rate-btn dislike"
            class:active={data.rating && data.rating.score === -1}
            onclick={() => rate(-1)}
          >
            ✕ Dislike <kbd>x</kbd>
          </button>
        </div>
      </section>

      {#if visibleCriteria.length}
        <section>
          <h3>Criteria</h3>
          <div class="chips">
            {#each visibleCriteria as c (c)}
              {@const cur = data.criteria ? data.criteria[c] : undefined}
              <span class="crit">
                <span class="crit-name">{c.replaceAll('_', ' ')}</span>
                <button class="chip mini" class:active={cur === -1} class:dislike={cur === -1} onclick={() => setCriterion(c, -1)}
                  >−</button
                >
                <button class="chip mini" class:active={cur === 1} class:like={cur === 1} onclick={() => setCriterion(c, 1)}
                  >+</button
                >
              </span>
            {/each}
          </div>
        </section>
      {/if}

      <section>
        <h3>Aspects</h3>
        <div class="chips">
          {#each $aspects as a (a.id)}
            <button class="chip" class:active={data.aspects && data.aspects.includes(a.id)} onclick={() => toggleAspect(a.id)}
              >{a.name}</button
            >
          {/each}
          {#if !$aspects.length}
            <span class="dim small">no aspects defined yet</span>
          {/if}
        </div>
      </section>

      <section>
        <h3>Critique {#if critiqueSaved}<span class="saved">saved ✓</span>{/if}</h3>
        <textarea
          class="ta"
          rows="3"
          placeholder="What is right / wrong with this image…"
          bind:value={critiqueText}
          oninput={critiqueChanged}
        ></textarea>
      </section>

      <section>
        <h3>Caption {#if captionSaved}<span class="saved">saved ✓</span>{/if}</h3>
        {#if autoTags.length}
          <div class="chips wrap">
            {#each autoTags as t, i (`${i}-${t}`)}
              <button class="chip mini" title="append to caption" onclick={() => appendTag(t)}>{t}</button>
            {/each}
          </div>
          {#if data.captions.auto.tagger}
            <div class="dim small" style="margin-bottom:6px">tagger: {data.captions.auto.tagger}</div>
          {/if}
        {/if}
        <textarea
          class="ta"
          rows="3"
          placeholder="Human caption for training…"
          bind:value={captionText}
          oninput={captionChanged}
        ></textarea>
        <div class="row">
          <button class="chip" onclick={autocaption} disabled={autocaptionPending}>
            {autocaptionPending ? 'auto-captioning…' : 'Auto-caption'}
          </button>
        </div>
        {#if data.captions && data.captions.refined}
          <div class="refined">
            <span class="lbl">refined</span>
            <p>{data.captions.refined.text}</p>
          </div>
        {/if}
      </section>

      <section>
        <h3>Metadata</h3>
        <dl class="meta">
          <dt>prompt</dt>
          <dd>
            {node.prompt || '—'}
            {#if node.prompt}<button class="chip mini ghost" onclick={copyPrompt}>copy</button>{/if}
          </dd>
          {#if node.positivePrefix}
            <dt>prefix</dt>
            <dd>
              <details><summary>positive prefix</summary><p>{node.positivePrefix}</p></details>
            </dd>
          {/if}
          {#if node.negativePrompt}
            <dt>negative</dt>
            <dd>
              <details><summary>negative prompt</summary><p>{node.negativePrompt}</p></details>
            </dd>
          {/if}
          <dt>model</dt>
          <dd>{modelLabel}</dd>
          {#if node.loraName}
            <dt>LoRA</dt>
            <dd>{node.loraName}</dd>
          {/if}
          <dt>seed</dt>
          <dd class="mono">{node.seed ?? '—'}</dd>
          <dt>steps / cfg</dt>
          <dd>{node.steps ?? '—'} / {node.cfg ?? '—'}</dd>
          <dt>sampler</dt>
          <dd>{node.samplerName ?? '—'} / {node.scheduler ?? '—'}</dd>
          {#if node.denoise != null}
            <dt>denoise</dt>
            <dd>{node.denoise}</dd>
          {/if}
          <dt>size</dt>
          <dd>{img.width}×{img.height}</dd>
          <dt>created</dt>
          <dd>{fmtDate(node.createdAt)}</dd>
          <dt>origin</dt>
          <dd>{node.origin || img.origin}{img.isImg2img ? ' · img2img' : ''}</dd>
          {#if img.sessionId != null}
            <dt>session</dt>
            <dd><button class="linklike" onclick={openSession}>open session in gallery</button></dd>
          {/if}
        </dl>
      </section>
    </div>
  </main>
{/if}
