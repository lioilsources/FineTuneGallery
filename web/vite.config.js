import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

const backend = 'http://localhost:8092';

export default defineConfig({
  plugins: [svelte()],
  build: {
    outDir: '../server/webdist',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/api': backend,
      '/img': backend,
      '/thumb': backend,
    },
  },
});
