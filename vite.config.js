import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

const projectRoot = path.dirname(fileURLToPath(import.meta.url));
const frontendRoot = path.resolve(projectRoot, 'static');

export default defineConfig({
  root: frontendRoot,
  plugins: [
    {
      name: 'aura-legacy-static-assets',
      transformIndexHtml: {
        order: 'pre',
        handler(html) {
          // The existing FastAPI page uses /static/... URLs. Vite's root is
          // already the static directory, so strip that prefix in the build.
          return html.replaceAll('/static/', '/');
        },
      },
      generateBundle() {
        // These legacy scripts are classic scripts rather than ES modules,
        // so emit them as standalone files in Vite's output directory.
        for (const filename of ['app.js', 'waterflow.js']) {
          this.emitFile({
            type: 'asset',
            fileName: filename,
            source: readFileSync(path.resolve(frontendRoot, filename)),
          });
        }
      },
    },
  ],
  build: {
    outDir: path.resolve(projectRoot, 'dist'),
    emptyOutDir: true,
  },
});
