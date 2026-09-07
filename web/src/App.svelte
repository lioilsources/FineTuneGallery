<script>
  import { route, matchRoute } from './lib/stores.js';
  import Gallery from './routes/Gallery.svelte';
  import Detail from './routes/Detail.svelte';
  import Datasets from './routes/Datasets.svelte';
  import DatasetDetail from './routes/DatasetDetail.svelte';
  import Stats from './routes/Stats.svelte';
  import Eval from './routes/Eval.svelte';
  import Toast from './components/Toast.svelte';

  const m = $derived(matchRoute($route));
</script>

<header class="topbar">
  <a class="brand" href="#/">FINETUNE<span>gallery</span></a>
  <nav class="topnav">
    <a class="chip" class:active={m.name === 'gallery' || m.name === 'image'} href="#/">Gallery</a>
    <a class="chip" class:active={m.name === 'datasets' || m.name === 'dataset'} href="#/datasets">Datasets</a>
    <a class="chip" class:active={m.name === 'eval'} href="#/eval">Eval</a>
    <a class="chip" class:active={m.name === 'stats'} href="#/stats">Stats</a>
  </nav>
</header>

{#if m.name === 'gallery'}
  <Gallery />
{:else if m.name === 'image'}
  <Detail id={m.id} />
{:else if m.name === 'datasets'}
  <Datasets />
{:else if m.name === 'dataset'}
  <DatasetDetail id={m.id} />
{:else if m.name === 'stats'}
  <Stats />
{:else if m.name === 'eval'}
  <Eval />
{/if}

<Toast />
