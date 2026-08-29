# Frontend

A single-file, no-build-step chat UI (`index.html`) for the Swastham-lite
API. Plain HTML/CSS/JS — no npm, no bundler, nothing to install.

## Design notes

Calm, "nightfall" palette (deep navy, soft lavender accent, warm amber
used sparingly) rather than a generic light/dark AI-chat template, since
the product is a sleep assistant. The one signature touch is a slow
"breathing" glow behind the header, echoing the rhythm of falling asleep
— restrained, and disabled automatically for users with
`prefers-reduced-motion` set.

Each assistant reply shows a small badge for which path produced it
(**Tool** / **Grounded answer** / **Needs more info** / **Support
resources**), so the UI doubles as a live demo of the agent's routing
logic, not just a black-box chat window. Cited sources are shown under
grounded answers.

## How to run

The backend must be running first:
```bash
cd ../src
uvicorn app:app --reload --port 8000
```

Then serve this folder (don't just double-click `index.html` — opening it
directly as a `file://` URL can trip browser CORS restrictions on the
`fetch()` calls). From this `frontend/` directory:

```bash
python -m http.server 5500
```

Then open **http://localhost:5500** in a browser.

If your API isn't running on `localhost:8000`, change the `API_BASE`
constant near the top of the `<script>` block in `index.html`.

## What it does

- Creates a session on load (`POST /session`)
- Sends each message to `POST /chat` and renders the response, including
  its routing mode and any cited sources
- Shows a connection status indicator, and a clear error state if the API
  isn't reachable
- A few suggestion chips to try each routing path (tool / RAG) without
  typing
