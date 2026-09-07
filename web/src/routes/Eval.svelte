<script>
  import { api } from '../lib/api.js';
  import {
    meta,
    sessions,
    aspects,
    ensureMeta,
    ensureAspects,
    ensureSessions,
    galleryFilter,
    emptyFilter,
    navigate,
  } from '../lib/stores.js';

  ensureMeta();
  ensureAspects();
  ensureSessions();

  const GROUPS = [
    ['model', 'Model'],
    ['style', 'Art style'],
    ['lora', 'LoRA'],
    ['pose', 'Pose'],
    ['session', 'Session'],
  ];

  let group = $state('model');
  let minRated = $state(10);
  let scope = $state({ model: '', session: '', style: '', aspect: '' });
  let sortCol = $state('n'); // 'n' | 'like' | <criterion>

  let data = $state(null);
  let loading = $state(false);
  let failed = $state(false);
  let generation = 0;

  function params() {
    const p = new URLSearchParams();
    p.set('group', group);
    p.set('min', String(minRated));
    for (const [k, v] of Object.entries(scope)) if (v) p.set(k, v);
    return p;
  }

  $effect(() => {
    const qs = params().toString();
    const gen = ++generation;
    loading = true;
    failed = false;
    api
      .get(`/api/eval?${qs}`)
      .then((res) => {
        if (gen !== generation) return;
        data = res;
      })
      .catch(() => {
        if (gen === generation) failed = true;
      })
      .finally(() => {
        if (gen === generation) loading = false;
      });
  });

  const criteria = $derived(data?.criteria ?? []);

  function cellOf(row, col) {
    return col === 'like' ? row.like : (row.criteria?.[col] ?? null);
  }

  // Rows sort by the selected column's Wilson lower bound. Cells with nothing
  // rated have no bound and sink to the bottom rather than reading as zero.
  const rows = $derived.by(() => {
    const list = [...(data?.rows ?? [])];
    if (sortCol === 'n') return list;
    return list.sort((a, b) => {
      const x = cellOf(a, sortCol);
      const y = cellOf(b, sortCol);
      const xl = x && x.lower !== null ? x.lower : -1;
      const yl = y && y.lower !== null ? y.lower : -1;
      if (xl !== yl) return yl - xl;
      return b.n - a.n;
    });
  });

  const pct = (v) => `${Math.round(v * 100)}%`;

  function cellState(cell) {
    if (!cell || cell.eligible === 0) return 'na'; // criterion cannot apply here
    if (cell.rated === 0) return 'unrated';
    if (cell.rated < (data?.minRated ?? 0)) return 'thin';
    return 'ok';
  }

  function cellTitle(row, col, cell) {
    if (!cell) return '';
    const name = col === 'like' ? 'likes' : col.replaceAll('_', ' ');
    if (cell.eligible === 0) return `${row.label}: no image here can be judged on ${name}`;
    if (cell.rated === 0) return `${row.label}: ${cell.eligible} image(s) eligible for ${name}, none rated yet`;
    return (
      `${row.label} · ${name}\n` +
      `${cell.up} up / ${cell.down} down of ${cell.eligible} eligible\n` +
      `rate ${pct(cell.rate)} · 95% interval ${pct(cell.lower)}–${pct(cell.upper)}` +
      (cell.rated < (data?.minRated ?? 0) ? `\nbelow the ${data.minRated}-rating floor — not ranked` : '')
    );
  }

  // Hand the unrated remainder of one cell to the gallery, pre-filtered. This
  // is the loop the harness exists to close: it says where the evidence is
  // thin, and the link takes you there to make it thicker.
  function rateGap(row, col) {
    const f = { ...emptyFilter };
    if (col === 'like') f.score = 'unrated';
    else f.criterion = `${col}:none`;
    if (group === 'model') f.model = row.key;
    else if (group === 'style') f.style = row.key;
    else if (group === 'session') f.session = row.key;
    else if (group === 'pose') f.pose = row.key || 'none';
    else if (group === 'lora') {
      const [name, strength] = row.key.split(' @ ');
      f.lora = row.key === '' ? 'none' : name;
      if (strength) f.loraStrength = strength;
    }
    // Scope filters set on this page carry over, so the queue is the same
    // slice the number came from.
    for (const [k, v] of Object.entries(scope)) if (v && !f[k]) f[k] = v;
    galleryFilter.set(f);
    navigate('/');
  }

  function resetScope() {
    scope = { model: '', session: '', style: '', aspect: '' };
  }

  const scoped = $derived(Object.values(scope).some(Boolean));
  const csvHref = $derived(`/api/eval?${params()}&format=csv`);

  const leaderList = $derived.by(() => {
    const l = data?.leaders ?? {};
    return ['like', ...criteria].filter((c) => l[c]).map((c) => [c, l[c]]);
  });
</script>

<main class="page">
  <h2>Eval harness</h2>
  <p class="dim small lede">
    Which slice of the corpus actually holds a pose, keeps the source identity, keeps the source style. Ranked by the
    lower bound of a 95% Wilson interval, so a perfect score on three images does not outrank a strong one on forty.
  </p>

  <div class="filterbar">
    <div class="filter-row">
      <span class="lbl">group by</span>
      <div class="seg" role="group" aria-label="Group by">
        {#each GROUPS as [val, label] (val)}
          <button
            class="seg-btn"
            class:active={group === val}
            onclick={(e) => {
              group = val;
              sortCol = 'n';
              e.currentTarget.blur();
            }}>{label}</button
          >
        {/each}
      </div>

      <span class="spacer"></span>

      <label class="lbl" for="min-rated" title="Groups with fewer ratings than this are shown but never crowned">
        min ratings
      </label>
      <input id="min-rated" class="inp num" type="number" min="0" max="999" bind:value={minRated} />
      <a class="chip mini ghost" href={csvHref} download>CSV</a>
    </div>

    <div class="filter-row">
      <span class="lbl">scope</span>
      <select class="sel" bind:value={scope.model} title="Restrict to one model">
        <option value="">All models</option>
        {#each $meta.models as m (m.id)}
          <option value={m.id}>{m.label}</option>
        {/each}
      </select>
      <select class="sel" bind:value={scope.session} title="Restrict to one session">
        <option value="">All sessions</option>
        {#each $sessions as s (s.id)}
          <option value={s.id}>{s.title} ({s.imageCount})</option>
        {/each}
      </select>
      {#if ($meta.styles || []).length}
        <select class="sel" bind:value={scope.style} title="Restrict to one art style">
          <option value="">All styles</option>
          {#each $meta.styles as st (st)}
            <option value={st}>{st}</option>
          {/each}
        </select>
      {/if}
      <select class="sel" bind:value={scope.aspect} title="Restrict to one aspect tag">
        <option value="">All aspects</option>
        {#each $aspects as a (a.id)}
          <option value={a.name}>{a.name}</option>
        {/each}
      </select>
      {#if scoped}
        <button class="chip ghost" onclick={resetScope}>clear scope</button>
      {/if}
    </div>
  </div>

  {#if failed}
    <p class="empty">Failed to load the eval matrix.</p>
  {:else if !data}
    <p class="dim">loading…</p>
  {:else if data.total.n === 0}
    <p class="empty">No images in this scope.</p>
  {:else}
    {#if leaderList.length}
      <h3>Leaders <span class="dim" style="text-transform:none">(min {data.minRated} ratings)</span></h3>
      <div class="tiles">
        {#each leaderList as [name, l] (name)}
          <div class="tile">
            <div class="tile-l">{name === 'like' ? 'likes' : name.replaceAll('_', ' ')}</div>
            <div class="leader-name" title={l.label}>{l.label}</div>
            <div class="leader-num">
              {pct(l.rate)}
              <span class="dim small">≥{pct(l.lower)} · n={l.rated}</span>
            </div>
            {#if l.behind}
              <div class="dim small">ahead of {l.behind}</div>
            {/if}
          </div>
        {/each}
      </div>
    {:else}
      <p class="empty">
        Nothing has reached {data.minRated} ratings yet. Rate more, or lower the floor to see provisional numbers.
      </p>
    {/if}

    <h3 style="margin-top: 22px">
      By {group}
      {#if loading}<span class="dim" style="text-transform:none">· refreshing</span>{/if}
    </h3>
    <p class="dim small key">
      <span class="key-mark"><span class="ivl-band-demo"></span></span>
      the bar is the 95% interval — a wide bar means thin evidence, not a bad model. Click a cell to go rate what is
      still missing.
    </p>

    <div class="tbl-scroll">
      <table class="tbl eval-tbl">
        <thead>
          <tr>
            <th>{group}</th>
            <th class="sortable" class:sorted={sortCol === 'n'}>
              <button class="th-btn" onclick={() => (sortCol = 'n')}>images</button>
            </th>
            {#each ['like', ...criteria] as col (col)}
              <th class="sortable" class:sorted={sortCol === col}>
                <button class="th-btn" onclick={() => (sortCol = col)}>
                  {col === 'like' ? 'likes' : col.replaceAll('_', ' ')}
                </button>
              </th>
            {/each}
          </tr>
        </thead>
        <tbody>
          {#each rows as row (row.key)}
            <tr>
              <td class="name" title={row.key || row.label}>{row.label}</td>
              <td class="num">
                {row.n}
                {#if row.unrated}<span class="dim small">· {row.unrated} unrated</span>{/if}
              </td>
              {#each ['like', ...criteria] as col (col)}
                {@const cell = cellOf(row, col)}
                {@const state = cellState(cell)}
                <td class="cell-eval">
                  {#if state === 'na'}
                    <span class="dim" title={cellTitle(row, col, cell)}>n/a</span>
                  {:else if state === 'unrated'}
                    <button class="gap" onclick={() => rateGap(row, col)} title={cellTitle(row, col, cell)}>
                      rate {cell.eligible}
                    </button>
                  {:else}
                    <button
                      class="cellbtn"
                      class:thin={state === 'thin'}
                      onclick={() => rateGap(row, col)}
                      title={cellTitle(row, col, cell)}
                    >
                      <span class="cell-top">
                        <span class="rate">{pct(cell.rate)}</span>
                        <span class="counts">
                          <span class="up">{cell.up}</span><span class="dim">/</span><span class="down">{cell.down}</span
                          >
                        </span>
                      </span>
                      <span class="ivl">
                        <span class="ivl-mid"></span>
                        <span class="ivl-band" style="left:{cell.lower * 100}%; width:{(cell.upper - cell.lower) * 100}%"
                        ></span>
                        <span class="ivl-tick" style="left:{cell.rate * 100}%"></span>
                      </span>
                      {#if cell.rated < cell.eligible}
                        <span class="cov dim">{cell.rated}/{cell.eligible} rated</span>
                      {/if}
                    </button>
                  {/if}
                </td>
              {/each}
            </tr>
          {/each}
        </tbody>
        <tfoot>
          <tr>
            <td class="name">all</td>
            <td class="num">{data.total.n}</td>
            {#each ['like', ...criteria] as col (col)}
              {@const cell = cellOf(data.total, col)}
              <td class="num">
                {#if cell && cell.rated}
                  {pct(cell.rate)} <span class="dim small">n={cell.rated}</span>
                {:else}
                  <span class="dim">—</span>
                {/if}
              </td>
            {/each}
          </tr>
        </tfoot>
      </table>
    </div>
  {/if}
</main>

<style>
  .lede {
    max-width: 70ch;
    margin: -6px 0 14px;
  }
  .key {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0 0 10px;
    max-width: 70ch;
  }
  .key-mark {
    flex: none;
    position: relative;
    display: inline-block;
    width: 54px;
    height: 6px;
    border-radius: 3px;
    background: var(--border);
  }
  .ivl-band-demo {
    position: absolute;
    left: 30%;
    width: 45%;
    top: 0;
    bottom: 0;
    border-radius: 3px;
    background: var(--accent);
    opacity: 0.55;
  }

  .leader-name {
    font-size: 15px;
    font-weight: 600;
    margin-top: 2px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .leader-num {
    font-size: 20px;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    display: flex;
    align-items: baseline;
    gap: 6px;
  }

  .eval-tbl td.name {
    max-width: 220px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .eval-tbl td.num {
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }
  .eval-tbl tfoot td {
    border-bottom: none;
    border-top: 1px solid var(--border);
    color: var(--text-dim);
  }

  .th-btn {
    background: none;
    border: 0;
    padding: 0;
    font: inherit;
    color: inherit;
    text-transform: inherit;
    letter-spacing: inherit;
    cursor: pointer;
  }
  .th-btn:hover {
    color: var(--text);
  }
  th.sorted .th-btn {
    color: var(--accent);
  }

  .cell-eval {
    min-width: 130px;
  }
  .cellbtn,
  .gap {
    display: block;
    width: 100%;
    background: none;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 3px 5px;
    text-align: left;
    font: inherit;
    color: var(--text);
    cursor: pointer;
  }
  .cellbtn:hover,
  .gap:hover {
    border-color: var(--border);
    background: var(--surface);
  }
  /* Below the ratings floor the number is real but not evidence — recede it
     rather than hide it, and let the wide interval bar do the arguing. */
  .cellbtn.thin .rate {
    color: var(--text-dim);
    font-weight: 500;
  }
  .gap {
    color: var(--text-dim);
    font-size: 12px;
  }
  .gap:hover {
    color: var(--accent);
  }

  .cell-top {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 8px;
  }
  .rate {
    font-size: 14px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }
  .counts {
    font-size: 11px;
  }
  .cov {
    display: block;
    font-size: 10.5px;
    margin-top: 1px;
  }

  .ivl {
    position: relative;
    display: block;
    height: 6px;
    margin: 3px 0 1px;
    border-radius: 3px;
    background: var(--border);
    overflow: hidden;
  }
  .ivl-mid {
    position: absolute;
    left: 50%;
    top: 0;
    bottom: 0;
    width: 1px;
    background: var(--bg);
    opacity: 0.7;
  }
  .ivl-band {
    position: absolute;
    top: 0;
    bottom: 0;
    min-width: 2px;
    border-radius: 3px;
    background: var(--accent);
    opacity: 0.5;
  }
  .ivl-tick {
    position: absolute;
    top: -1px;
    bottom: -1px;
    width: 2px;
    margin-left: -1px;
    border-radius: 1px;
    background: var(--accent);
  }
  .cellbtn.thin .ivl-band,
  .cellbtn.thin .ivl-tick {
    background: var(--text-dim);
  }
</style>
