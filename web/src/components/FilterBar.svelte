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
    f.model !== '' ||
      f.aspect !== '' ||
      f.score !== 'all' ||
      f.session !== '' ||
      f.style !== '' ||
      f.medium !== '' ||
      f.translator !== '' ||
      f.lora !== '' ||
      f.pose !== '' ||
      f.uncaptioned ||
      f.criterion !== '' ||
      f.judge !== '' ||
      f.taxo !== '' ||
      f.tempMax < 1 ||
      f.shared ||
      f.ratingUnsure ||
      f.labelQueue
  );

  /* ---------------- content taxonomy ---------------- */

  const taxo = $derived($meta.taxonomy || { enabled: false });
  const taxoPairs = $derived(f.taxo ? f.taxo.split(',') : []);

  function bandFor(t) {
    const th = taxo.thresholds;
    if (!th) return '';
    if (t <= th.safe_max) return 'safe';
    if (t <= th.suggestive_max) return 'suggestive';
    if (t <= th.nsfw_max) return 'nsfw';
    return 'explicit';
  }

  function addTaxo(pair) {
    if (!pair || taxoPairs.includes(pair)) return;
    // one value per single-valued node: picking brown replaces blonde
    const node = pair.split(':')[0];
    const facet = (taxo.facets || []).find((x) => x.id === node);
    const keep = facet && facet.kind === 'multi' ? taxoPairs : taxoPairs.filter((p) => !p.startsWith(node + ':'));
    setF({ taxo: [...keep, pair].join(',') });
  }

  function dropTaxo(pair) {
    setF({ taxo: taxoPairs.filter((p) => p !== pair).join(',') });
  }

  function facetLabel(id) {
    const fc = (taxo.facets || []).find((x) => x.id === id);
    return fc && fc.labelCs ? fc.labelCs : id;
  }

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
    if (f.style) parts.push(`style: ${f.style}`);
    if (f.medium) parts.push(`medium: ${f.medium}`);
    if (f.translator) parts.push(`translator: ${f.translator === 'none' ? 'raw prose' : f.translator}`);
    if (f.lora) parts.push(`lora: ${f.lora === 'none' ? 'none' : f.lora}${f.loraStrength ? ` @ ${f.loraStrength}` : ''}`);
    if (f.pose) parts.push(`pose: ${f.pose}`);
    if (f.uncaptioned) parts.push('uncaptioned');
    if (f.criterion)
      parts.push(f.criterion.replace(':1', ' +').replace(':-1', ' −').replace(':none', ' — unrated'));
    if (f.judge)
      parts.push(
        f.judge.replace(':disagree', ' — judge disagrees').replace(':unrated', ' — judged, not rated by you')
      );
    if (f.shared) parts.push('shared view (safe only)');
    if (f.tempMax < 1) parts.push(`teplota ≤ ${f.tempMax.toFixed(2)}`);
    for (const pr of f.taxo ? f.taxo.split(',') : []) parts.push(pr.replace(':', ' = '));
    if (f.ratingUnsure) parts.push('rating unsure');
    if (f.labelQueue) parts.push('label queue');
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
        <!-- The GPU box cannot load it right now, but its images are still
             here and still worth filtering to — hence listed, marked. -->
        <option value={mm.id}>{mm.label}{mm.available ? '' : ' (offline)'}</option>
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

    <!-- Only rendered once styled images exist: on a gallery fed purely by the
         phone this dropdown would be an empty control. -->
    {#if ($meta.styles || []).length}
      <select class="sel" value={f.style} onchange={(e) => setF({ style: e.target.value })} title="Art style">
        <option value="">All styles</option>
        {#each $meta.styles as st (st)}
          <option value={st}>{st}</option>
        {/each}
      </select>
    {/if}

    <select class="sel" value={f.criterion} onchange={(e) => setF({ criterion: e.target.value })} title="Criterion filter">
      <option value="">Any criterion</option>
      {#each $meta.criteria as c (c)}
        <option value="{c}:1">{c} +</option>
        <option value="{c}:-1">{c} −</option>
        <option value="{c}:none">{c} — unrated</option>
      {/each}
    </select>

    <!-- VL judge queues. disagree = the labels the next prompt version is
         written from; unrated = the blind calibration queue (no verdict shown
         until you rate). Hidden when no judge is configured. -->
    {#if $meta.judge?.enabled}
      <select class="sel" value={f.judge} onchange={(e) => setF({ judge: e.target.value })} title="VL judge queue">
        <option value="">Any judge state</option>
        {#each $meta.criteria as c (c)}
          <option value="{c}:disagree">{c} — judge disagrees</option>
          <option value="{c}:unrated">{c} — judged, rate blind</option>
        {/each}
      </select>
    {/if}

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

  <!-- Content taxonomy: hidden unless the server runs with FINETUNE_TAXONOMY=1.
       Minor age groups are not in the facets (the server refuses them too). -->
  {#if taxo.enabled}
    <div class="filter-row">
      <span class="lbl">obsah</span>
      <label class="temp" title="Teplota obsahu 0–1 (WD14 + 2 NSFW klasifikátory). 1 = bez limitu.">
        teplota ≤
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={f.tempMax}
          disabled={taxo.publicView}
          onchange={(e) => setF({ tempMax: Number(e.currentTarget.value) })}
          onkeydown={(e) => e.stopPropagation()}
        />
        <span class="temp-val">{f.tempMax < 1 ? `${f.tempMax.toFixed(2)} · ${bandFor(f.tempMax)}` : 'vše'}</span>
      </label>
      <button
        class="chip"
        class:active={f.shared || taxo.publicView}
        disabled={taxo.publicView}
        title="Jako sdílený / veřejný pohled: jen safe a jisté"
        onclick={(e) => {
          setF({ shared: !f.shared });
          e.currentTarget.blur();
        }}>shared</button
      >
      <select
        class="sel"
        value=""
        onchange={(e) => {
          addTaxo(e.currentTarget.value);
          e.currentTarget.value = '';
        }}
        title="Facety taxonomie"
      >
        <option value="">+ facet…</option>
        {#each taxo.facets || [] as fc (fc.id)}
          <optgroup label={fc.labelCs || fc.id}>
            {#each fc.values as v (v.id)}
              <option value="{fc.id}:{v.id}">{v.id} ({v.count}){fc.humanRequired ? ' · ručně' : ''}</option>
            {/each}
          </optgroup>
        {/each}
      </select>
      {#each taxoPairs as pr (pr)}
        <button class="chip active" title="odebrat" onclick={() => dropTaxo(pr)}
          >{facetLabel(pr.split(':')[0])}: {pr.split(':')[1]} ×</button
        >
      {/each}
      <button
        class="chip"
        class:active={f.ratingUnsure}
        title="Hodnotitelé teploty se neshodli — fronta pro člověka"
        onclick={(e) => {
          setF({ ratingUnsure: !f.ratingUnsure });
          e.currentTarget.blur();
        }}>unsure ({taxo.unsure || 0})</button
      >
      <button
        class="chip"
        class:active={f.labelQueue}
        title="Vzorek k ručnímu olabelování (naslepo — odpovědi hodnotitelů se ukážou až po labelu)"
        onclick={(e) => {
          setF({ labelQueue: !f.labelQueue });
          e.currentTarget.blur();
        }}>label queue</button
      >
    </div>
  {/if}

  <div class="filter-row summary-row">
    <span>{$galleryState.items.length} loaded{$galleryState.nextCursor ? '+' : ''}</span>
    {#if summary}
      <span>·</span>
      <span class="summary">{summary}</span>
    {/if}
    <span class="spacer"></span>
    <a class="chip mini ghost" href="#/datasets">datasets</a>
    <a class="chip mini ghost" href="#/eval">eval</a>
    <a class="chip mini ghost" href="#/stats">stats</a>
  </div>
</div>
