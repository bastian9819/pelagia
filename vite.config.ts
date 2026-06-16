import { defineConfig } from 'vitest/config';

// Relative base so the static build works both at a domain root and under a
// GitHub Pages project subpath (e.g. /pelagia/).
export default defineConfig({
  base: './',
  build: {
    target: 'es2022',
  },
  test: {
    // The simulation core is pure logic; it runs and is tested headless in Node.
    environment: 'node',
    include: ['test/**/*.test.ts', 'src/**/*.test.ts'],
  },
});
