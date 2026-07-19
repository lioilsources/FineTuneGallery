<script>
  import { get } from 'svelte/store';
  import { meta, aspects, ensureMeta, ensureAspects, toast } from '../lib/stores.js';
  import { api } from '../lib/api.js';

  let datasets = $state(null);

  async function load() {
    try {
      datasets = await api.get('/api/datasets');
    } catch {
      datasets = datasets || [];
    }
  }
  load();

  const trainable = $derived(($meta.models || []).filter((m) => m.trainable));
  const aspectById = $derived(new Map($aspects.map((a) => [a.id, a])));

  /* ------- new dataset form ------- */

  let name = $state('');
  let aspectId = $state('');
  let baseModel = $state('');
  let triggerWord = $state('');
  let repeats = $state(10);
  let creating = $state(false);
  let triggerTouched = false;

  Promise.all([ensureMeta(), ensureAspects()]).then(() => {
    const list = get(aspects);
    if (!aspectId && list.length) {
      aspectId = String(list[0].id);
      if (!triggerTouched && !triggerWord) triggerWord = list[0].name;
    }
    const t = (get(meta).models || []).filter((m) => m.trainable);
    if (!baseModel && t.length) baseModel = t[0].id;
  });

  function aspectNameFor(idVal) {
    const a = aspectById.get(Number(idVal));
    return a ? a.name : '';
  }

  function onAspectChange() {
    if (!triggerTouched) triggerWord = aspectNameFor(aspectId);
  }

  async function create() {
    if (!name.trim() || !aspectId || !baseModel) {
      toast.error('Name, aspect and base model are required');
      return;
    }
    creating = true;
    try {
      const ds = await api.post('/api/datasets', {
        name: name.trim(),
        aspectId: Number(aspectId),
        baseModel,
        triggerWord: triggerWord.trim() || aspectNameFor(aspectId),
        repeats: Number(repeats) || 10,
      });
      toast.info(`Dataset "${ds.name}" created`);
      name = '';
      triggerTouched = false;
      await load();
    } catch {
      /* toasted by api */
    } finally {
      creating = false;
    }
  }

  async function remove(d) {
    if (!confirm(`Delete dataset "${d.name}"?`)) return;
    try {
      await api.del(`/api/datasets/${d.id}`);
      toast.info(`Dataset "${d.name}" deleted`);
      await load();
    } catch {
      /* toasted by api */
    }
  }

  function isBuilt(d) {
    return d.status === 'built' || !!d.builtAt;
  }

  function fmtDate(s) {
    if (!s) return '—';
    const d = new Date(s);
    return isNaN(d.getTime()) ? s : d.toLocaleString();
  }
</script>

<main class="page">
  <h2>Datasets</h2>

  <div class="card">
    <div class="form-row">
      <input class="inp" placeholder="dataset name" bind:value={name} />
      <select class="sel" bind:value={aspectId} onchange={onAspectChange} title="Aspect">
        {#each $aspects as a (a.id)}
          <option value={String(a.id)}>{a.name}</option>
        {/each}
      </select>
      <div class="field">
        <select class="sel" bind:value={baseModel} title="Base model (trainable only)">
          {#each trainable as m (m.id)}
            <option value={m.id}>{m.label} ({m.ecosystem})</option>
          {/each}
        </select>
        <p class="hint">
          LoRA works within the model's ecosystem (e.g. pony ↔ atomix). Training on liked images from another model =
          deliberate cross-model distillation.
        </p>
      </div>
      <input class="inp" placeholder="trigger word" bind:value={triggerWord} oninput={() => (triggerTouched = true)} />
      <input class="inp num" type="number" min="1" bind:value={repeats} title="repeats" />
      <button class="btn" onclick={create} disabled={creating}>{creating ? 'creating…' : 'New dataset'}</button>
    </div>
  </div>

  <div style="height: 16px"></div>

  {#if datasets === null}
    <p class="dim">loading…</p>
  {:else if !datasets.length}
    <p class="empty">No datasets yet — create one above.</p>
  {:else}
    <div class="tbl-scroll">
      <table class="tbl">
        <thead>
          <tr>
            <th>name</th>
            <th>aspect</th>
            <th>base model</th>
            <th>trigger</th>
            <th>repeats</th>
            <th>status</th>
            <th>items</th>
            <th>built</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {#each datasets as d (d.id)}
            <tr>
              <td><a href={`#/datasets/${d.id}`}>{d.name}</a></td>
              <td>{aspectById.get(d.aspectId)?.name ?? d.aspectId}</td>
              <td>{d.baseModel}</td>
              <td>{d.triggerWord}</td>
              <td>{d.repeats}</td>
              <td><span class="status s-{d.status}">{d.status}</span></td>
              <td>{d.itemCount}</td>
              <td class="dim">{fmtDate(d.builtAt)}</td>
              <td>
                <span class="actions">
                  <a class="chip mini" href={`#/datasets/${d.id}`}>open</a>
                  {#if isBuilt(d)}
                    <a class="chip mini" href={`/api/datasets/${d.id}/download`}>download</a>
                  {/if}
                  <button class="chip mini ghost" onclick={() => remove(d)}>delete</button>
                </span>
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
</main>
