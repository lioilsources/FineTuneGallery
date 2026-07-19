<script>
  let { item, index, selected = false, onselect, onopen } = $props();

  // First click selects; clicking the already-selected cell opens it
  // (so two quick clicks = open, and keyboard users can click-open directly).
  function click() {
    if (selected) onopen(item);
    else onselect(index);
  }
</script>

<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div class="cell" class:selected data-idx={index} onclick={click} title={item.prompt || ''}>
  <img
    class="cell-img"
    src="/thumb/{item.sha256}"
    alt={item.prompt ? item.prompt.slice(0, 60) : `image ${item.id}`}
    loading="lazy"
    onerror={(e) => e.currentTarget.classList.add('broken')}
  />
  <div class="cell-badges">
    {#if item.score === 1}<span class="badge b-like">♥</span>{/if}
    {#if item.score === -1}<span class="badge b-dislike">✕</span>{/if}
    {#if item.aspects && item.aspects.length}<span class="badge b-aspect">{item.aspects.length}</span>{/if}
    {#if item.isImg2img}<span class="badge b-flag">i2i</span>{/if}
    {#if item.origin === 'upload'}<span class="badge b-flag">up</span>{/if}
    {#if item.poseId}<span class="badge b-pose"><img src="/poses/{item.poseId}.png" alt="pose {item.poseId}" /></span>{/if}
  </div>
</div>
