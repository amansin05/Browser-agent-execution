# Vendored: Mozilla Readability

Standalone copy of [mozilla/readability](https://github.com/mozilla/readability) — the algorithm
behind Firefox Reader Mode. We inject `Readability.js` into the live page via the Playwright MCP
`browser_evaluate` tool to turn an opaque/JS-heavy page into clean, LLM-readable text (see
`browser_agent/agent/reader.py`). This replaced the old screenshot/vision fallback.

- **Source:** https://github.com/mozilla/readability
- **Commit:** `08be6b4bdb204dd333c9b7a0cfbc0e730b257252`
- **License:** Apache-2.0 (see `LICENSE.md`)

Files are vendored **unmodified**. `Readability.js` declares a top-level `function Readability(...)`
and guards its `module.exports` with `typeof module === "object"`, so injecting the whole source into
a page-context function body is safe (`module` is undefined in the browser, so the export is skipped)
and `new Readability(document.cloneNode(true)).parse()` works directly.

To update: re-clone upstream, copy `Readability.js` + `Readability-readerable.js` + `LICENSE.md` here,
and bump the commit hash above.
