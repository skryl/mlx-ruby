# GitHub Pages Docs + Web Demo Deploy RFP

## Status

Completed (2026-02-19)

## Context

GitHub Pages deployment currently publishes only documentation HTML. The project
also has a browser demo site under `web/` that should be published alongside
the docs. We need a single Pages artifact where docs stay at the site root and
the demo is available under a dedicated subpath.

## Goals

1. Publish docs as the top-level GitHub Pages site (`/`).
2. Publish the web demo under a stable subpath (`/demo/`).
3. Keep deployment automated through the existing documentation workflow.
4. Add a README link to the deployed web demo.

## Non-goals

1. Reworking demo runtime logic or model export scripts.
2. Changing docs information architecture.
3. Adding new demo pages or features.

## Phased Plan (Red/Green)

### Phase 1: Confirm current deployment gap

Red:
1. Verify deployment action uploads only `docs/build/html`.
2. Verify workflow path/steps do not stage `web/` into Pages artifact.

Green:
1. Capture baseline references in workflow/action files for traceability.

Refactor:
1. Keep baseline checks documented in this RFP only (no code changes yet).

Exit criteria:
1. Baseline evidence confirms web demo is not currently published by Pages.

### Phase 2: Build unified Pages artifact

Red:
1. Add/prepare staging flow for docs + web and run syntax checks.
2. Confirm old artifact path assumptions fail after staging switch (expected).

Green:
1. Update Pages build action to:
   - build docs
   - assemble a combined site directory
   - copy docs HTML to root
   - copy `web/` to `/demo/`
   - upload the combined directory as Pages artifact
2. Ensure docs `index.html` remains root entrypoint.

Refactor:
1. Exclude unnecessary files from copied web artifact (for example local
   dependency directories) to keep artifact size controlled.

Exit criteria:
1. Build action emits a single combined Pages artifact.
2. Combined artifact contains docs root and `demo/index.html`.

### Phase 3: Wire workflow + README entrypoint

Red:
1. Verify workflow trigger scope does not account for web demo changes.
2. Verify README has no GitHub Pages link for web demo.

Green:
1. Update workflow trigger/config as needed for combined deployment.
2. Add README index link to deployed web demo path.

Refactor:
1. Keep link naming and path conventions aligned with existing docs links.

Exit criteria:
1. Workflow is configured for combined docs+demo Pages deployment.
2. README includes the public demo link.

## Acceptance Criteria

1. GitHub Pages artifact includes docs at root and demo at `/demo/`.
2. Top-level site index remains docs `index.html`.
3. Web demo index is reachable at `/demo/`.
4. README includes a clear link to the deployed web demo.

## Risks and Mitigations

1. Risk: Demo copy includes local development dependencies and inflates Pages
   artifact size.
   Mitigation: explicitly exclude `web/node_modules` and transient files.
2. Risk: Workflow trigger misses relevant file changes.
   Mitigation: broaden trigger scope to include demo/deploy config paths.
3. Risk: Path breakage under subpath deployment.
   Mitigation: preserve relative links by copying full `web/` tree under a
   fixed subdirectory.

## Implementation Checklist

- [x] Phase 1 red: baseline deployment scope captured.
- [x] Phase 2 green: build action stages docs + web demo in one artifact.
- [x] Phase 2 refactor: unnecessary web files excluded from artifact.
- [x] Phase 3 green: workflow updated for combined deployment.
- [x] Phase 3 green: README link to deployed `/demo/` added.
- [x] Finalize status to `Completed` with date after verification gates pass.
