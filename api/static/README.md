# Embedding the visual search widget on decorurs.com

`decorurs-widget.js` is a single, dependency-free JS file: a small round
icon in the bottom-right corner that opens a chat panel on the right where
a shopper can type or speak a description ("a round marble coffee table")
or upload a photo, and get back the closest-matching products.

It's served by the API itself at `/widget/decorurs-widget.js` (see the
`StaticFiles` mount in `api/main.py`), so **one deployment gives you both
the search API and the script your storefront loads** — no separate static
host needed.

## 0. Try it locally first

```bash
docker compose up -d --build
docker compose run --rm indexer   # if you haven't indexed the catalog yet
```

Open **http://localhost:8000/widget/demo.html** — that page stands in for
decorurs.com and loads the real widget against your local API. Confirm the
icon appears, a typed query returns results, an uploaded photo returns
results, and (in Chrome/Edge) the mic button works.

## 1. Deploy the API somewhere with a public HTTPS URL

decorurs.com is a Shopify store, which is always served over HTTPS —
browsers block a script on an HTTPS page from calling an HTTP API (mixed
content), so the API needs a real TLS certificate, not just an IP:port.
The `docker-compose.yml` in this repo (qdrant + api, `indexer` run as a
one-off job) will run as-is on any Docker host. A few common options:

- **Railway / Render / Fly.io** — point them at this repo, they build from
  the Dockerfiles directly and give you a `https://...` URL for the `api`
  service for free/cheap.
- **A VM (DigitalOcean, EC2, etc.)** — run `docker compose up -d --build`
  on the box, put a reverse proxy (Caddy or nginx + certbot) in front of
  the `api` service for a real certificate, e.g. `api.decorurs.com`.

Whichever you choose, you need the `api` service reachable at a stable
HTTPS domain — call it `https://api.decorurs.com` below.

**Set CORS before you deploy.** Shopify pages run on `https://decorurs.com`
(and `https://www.decorurs.com` if that's used too), so the API must allow
those specific origins:

```bash
# .env
CORS_ORIGINS=https://decorurs.com,https://www.decorurs.com
```

## 2. Index the catalog against that deployment

Same as local setup, just run once against the deployed stack:

```bash
docker compose run --rm indexer
```

Re-run it any time products are added, removed, or re-photographed.

## 3. Add the widget to the Shopify theme

In the Shopify admin:

**Online Store → Themes → (your live theme) → Edit code → `layout/theme.liquid`**

Paste this immediately before the closing `</body>` tag:

```html
<script
  src="https://api.decorurs.com/widget/decorurs-widget.js"
  data-api-url="https://api.decorurs.com"
  defer
></script>
```

Replace `https://api.decorurs.com` in **both** places with wherever you
deployed the API in step 1. Save. The icon should now appear on every page
of the live site (theme.liquid wraps every template).

If you'd rather not edit `theme.liquid` directly, the same snippet also
works dropped into a **Custom HTML** section/block on specific pages, or
via **Online Store → Themes → Edit code → app embed** if your theme
supports app embeds — the script tag itself doesn't change either way.

## 4. Verify on the live site

Open decorurs.com in a real browser (not the Shopify theme editor preview,
which runs on a different origin than your CORS allowlist) and confirm the
icon shows up bottom-right, and that both a typed query and an uploaded
photo return results. Check the browser console for any CORS or
network errors if not — the most common cause is the deployed origin not
matching what's in `CORS_ORIGINS`.

## Optional configuration

Set these before the widget script tag loads, if you want to override the
defaults:

```html
<script>
  window.DecorUrsVisualSearchConfig = {
    apiUrl: "https://api.decorurs.com",   // alternative to data-api-url
    contactEmail: "alok@trustic.ca",       // shown when a search has no matches
    contactPhone: "780-604-5390",
    speechLang: "en-US",                   // BCP-47 tag for the mic's speech recognition
  };
</script>
<script src="https://api.decorurs.com/widget/decorurs-widget.js" defer></script>
```

## Notes

- The widget is pure vanilla JS/CSS in a single file — no build step, no
  npm install, so it drops into Shopify's theme editor as-is.
- All of its styles are scoped under `#dvs-root`, and it wraps its markup
  in `all: initial` before applying its own font/colors, so it shouldn't
  visually collide with the theme's CSS. If something still looks off
  next to a particular theme, it's almost always a `z-index` fight —
  the widget uses `2147483000`+, about as high as z-index reasonably goes.
- The mic button only appears in browsers that support the Web Speech
  API (Chrome, Edge, Safari with limits; not Firefox) — it's hidden
  automatically everywhere else, so text and photo search always work
  regardless of browser.
- This is separate from `frontend/` (the standalone Next.js search page).
  You can keep, redeploy, or remove that microsite independently — the
  widget only depends on the `api` service.
