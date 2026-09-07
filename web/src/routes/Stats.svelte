<script>
  import { api } from '../lib/api.js';

  let stats = $state(null);
  let error = $state(false);

  (async () => {
    try {
      stats = await api.get('/api/stats');
    } catch {
      error = true;
    }
  })();
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

    <h3 style="margin-top: 24px">Model × criterion</h3>
    <p class="dim small">
      Moved to the <a href="#/eval">eval harness</a>, which reports the same tallies with sample size and a confidence
      interval attached — the counts alone rank a lucky 3/3 above a solid 36/40.
    </p>
  {/if}
</main>
