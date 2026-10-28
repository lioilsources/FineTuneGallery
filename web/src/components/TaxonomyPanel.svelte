<script>
  // Content taxonomy for one image: the three evaluators, their consensus, the
  // temperature — and the human label, which wins. In the labelling sample the
  // server returns no evaluator answer until the human has labelled (blind,
  // like the judge), so κ measures agreement, not deference.
  import { api } from '../lib/api.js';
  import { toast } from '../lib/stores.js';

  let { id } = $props();

  let data = $state(null);
  let similar = $state([]);
  let draft = $state({}); // node → value | [values]
  let rating = $state('');
  let saving = $state(false);
  let gen = 0;

  const EVALS = ['consensus', 'wd14', 'siglip', 'vlm', 'head'];

  $effect(() => {
    load(id);
  });

  async function load(imageId) {
    const g = ++gen;
    data = null;
    similar = [];
    try {
      const d = await api.get(`/api/images/${imageId}/taxonomy`);
      if (g !== gen) return;
      data = d;
      const h = d.human || {};
      const next = {};
      for (const n of d.nodes) {
        const vals = (h[n.id] && h[n.id].values) || [];
        next[n.id] = n.kind === 'multi' ? [...vals] : vals[0] || '';
      }
      draft = next;
      rating = (h.rating && h.rating.values && h.rating.values[0]) || '';
    } catch {
      return;
    }
    try {
      const s = await api.get(`/api/images/${imageId}/similar?limit=12`);
      if (g === gen) similar = s.items || [];
    } catch {
      /* no embedding yet — the strip just stays empty */
    }
  }

  function answer(ev, nodeId) {
    const a = data && data.evaluators && data.evaluators[ev] && data.evaluators[ev][nodeId];
    if (!a) return null;
    return a;
  }

  function fmt(a) {
    if (!a) return '—';
    const v = a.values.length ? a.values.join(', ') : '∅';
    return a.unsure ? `${v} ?` : v;
  }

  function toggleMulti(nodeId, v) {
    const cur = draft[nodeId] || [];
    draft = { ...draft, [nodeId]: cur.includes(v) ? cur.filter((x) => x !== v) : [...cur, v] };
  }

  async function save() {
    saving = true;
    const labels = {};
    for (const n of data.nodes) {
      const v = draft[n.id];
      const empty = n.kind === 'multi' ? false : !v;
      labels[n.id] = empty ? null : v;
    }
    try {
      await api.put(`/api/images/${id}/taxonomy`, { labels, rating: rating || null });
      toast.info('Taxonomy labels saved');
      await load(id);
    } catch {
      /* toasted by api */
    } finally {
      saving = false;
    }
  }
</script>

{#if data}
  <section class="taxo-panel">
    <h3>
      Obsah
      {#if data.rating}
        · teplota {data.rating.temperature?.toFixed(2)} <span class:unsure={data.rating.unsure}>{data.rating.band}{data.rating.unsure ? ' (unsure)' : ''}</span>
      {/if}
      {#if data.blind}<span class="dim">· naslepo (label queue)</span>{/if}
    </h3>
    <table>
      <thead>
        <tr>
          <th>uzel</th>
          {#if !data.blind}
            {#each EVALS as ev (ev)}<th>{ev}</th>{/each}
          {/if}
          <th>ty</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>rating</td>
          {#if !data.blind}
            <td>{data.rating ? data.rating.band : '—'}</td>
            <td colspan="2"></td>
            <td>{fmt(answer('vlm', 'rating'))}</td>
            <td></td>
          {/if}
          <td>
            <select class="sel" bind:value={rating}>
              <option value="">—</option>
              {#each data.bands as b (b)}<option value={b}>{b}</option>{/each}
            </select>
          </td>
        </tr>
        {#each data.nodes as n (n.id)}
          <tr>
            <td title={n.id}>{n.labelCs || n.id}</td>
            {#if !data.blind}
              {#each EVALS as ev (ev)}
                {@const a = answer(ev, n.id)}
                <td class:unsure={a && a.unsure}>{fmt(a)}</td>
              {/each}
            {/if}
            <td>
              {#if n.kind === 'multi'}
                <div class="multi">
                  {#each n.values as v (v.id)}
                    <button
                      class="chip mini"
                      class:active={(draft[n.id] || []).includes(v.id)}
                      onclick={() => toggleMulti(n.id, v.id)}>{v.id}</button
                    >
                  {/each}
                </div>
              {:else}
                <select class="sel" bind:value={draft[n.id]}>
                  <option value="">—</option>
                  {#each n.values as v (v.id)}<option value={v.id}>{v.id}</option>{/each}
                </select>
              {/if}
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
    <button class="chip" onclick={save} disabled={saving}>{saving ? 'saving…' : 'save labels'}</button>
    {#if similar.length}
      <div class="taxo-similar">
        {#each similar as s (s.id)}
          <a href={`#/image/${s.id}`} title={`cos ${s.score}`}><img src="/thumb/{s.sha256}" alt="similar" /></a>
        {/each}
      </div>
    {/if}
  </section>
{/if}
