// Legacy compatibility shim.
// Production baseball data is now fetched directly by run_i2_today.mjs from the
// approved source hierarchy (MLB Stats API plus approved public/explicit sources).
// This file intentionally contains no Netlify/upstream proxy behavior.
await import('./run_i2_today.mjs');
