// Compatibility entry point. Source selection and exact identity resolution now live
// in the common production runner. Legacy unverified lineup overrides are not trusted.
await import('./run_i2_today_upstream_wrapper.mjs');
