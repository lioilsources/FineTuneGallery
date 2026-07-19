<script>
  import { toasts } from '../lib/stores.js';
</script>

{#if $toasts.length}
  <div class="toasts">
    {#each $toasts as t (t.id)}
      <div class="toast {t.kind}">{t.msg}</div>
    {/each}
  </div>
{/if}
