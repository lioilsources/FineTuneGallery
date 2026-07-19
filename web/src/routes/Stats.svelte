<script>
  import { meta, ensureMeta } from '../lib/stores.js';
  import { api } from '../lib/api.js';

  ensureMeta();

  let stats = $state(null);
  let error = $state(false);

  (async () => {
    try {
      stats = await api.get('/api/stats');
    } catch {
      error = true;
    }
  })();

  const modelRows = $derived.by(() => {
    if (!stats) return [];
    const ids = [];
    for (const e of stats.byModelCriterion || []) {
      if (!ids.includes(e.modelId)) ids.push(e.modelId);
    }
    return ids;
  });

  const critCols = $derived.by(() => {
    const cols = [...($meta.criteria || [])];
    for (const e of (stats && stats.byModelCriterion) || []) {
      if (!cols.includes(e.criterion)) cols.push(e.criterion);
    }
    return cols;
  });

  const modelLabels = $derived(new Map(($meta.models || []).map((m) => [m.id, m.label])));

  function cell(modelId, criterion) {
    return (stats.byModelCriterion || []).find((e) => e.modelId === modelId && e.criterion === criterion) || null;
  }
</script>

<main class="page">
  <h2>Stats</h2>

  {#if error}
    <p class="empty">Failed to load stats.</p>
  {:else if !stats}
    <p class="dim">loading…</p>
  {:else}
    <div class="tiles">
      <div class="tile">
        <div class="tile-n">{stats.total}</div>
        <div class="tile-l">total images</div>
      </div>
      <div class="tile">
        <div class="tile-n">{stats.unrated}</div>
        <div class="tile-l">unrated backlog</div>
      </div>
      <div class="tile">
        <div class="tile-n like-t">{stats.liked}</div>
        <div class="tile-l">♥ liked</div>
      </div>
      <div class="tile">
        <div class="tile-n dislike-t">{stats.disliked}</div>
        <div class="tile-l">✕ disliked</div>
      </div>
    </div>

    <h3>By aspect</h3>
    {#if stats.byAspect && stats.byAspect.length}
      <div class="tbl-scroll">
        <table class="tbl">
          <thead>
            <tr><th>aspect</th><th>total</th><th>liked</th></tr>
          </thead>
          <tbody>
            {#each stats.byAspect as a (a.name)}
              <tr>
                <td>{a.name}</td>
                <td>{a.total}</td>
                <td class="up">{a.liked}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {:else}
      <p class="dim small">no aspect data yet</p>
    {/if}

    <h3 style="margin-top: 24px">By model × criterion <span class="dim" style="text-transform:none">(up / down)</span></h3>
    {#if modelRows.length}
      <div class="tbl-scroll">
        <table class="tbl">
          <thead>
            <tr>
              <th>model</th>
              {#each critCols as c (c)}
                <th>{c.replaceAll('_', ' ')}</th>
              {/each}
            </tr>
          </thead>
          <tbody>
            {#each modelRows as mid (mid)}
              <tr>
                <td>{modelLabels.get(mid) ?? mid}</td>
                {#each critCols as c (c)}
                  {@const e = cell(mid, c)}
                  <td>
                    {#if e}
                      <span class="up">{e.up}</span><span class="dim">/</span><span class="down">{e.down}</span>
                    {:else}
                      <span class="dim">—</span>
                    {/if}
                  </td>
                {/each}
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {:else}
      <p class="dim small">no criterion data yet</p>
    {/if}
  {/if}
</main>
