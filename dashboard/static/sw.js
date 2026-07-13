/* WEEX Bot Command Center — lightweight PWA service worker */
const CACHE = "weex-cc-v3";
const PRECACHE = [
  "/",
  "/static/styles.css",
  "/static/app.js",
  "/static/logo.svg",
  "/static/manifest.webmanifest",
  "/static/icons/icon-192.png",
  "/static/icons/icon-512.png",
  "/static/icons/apple-touch-icon.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll(PRECACHE)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;

  const url = new URL(req.url);

  // Always network-first for live API data
  if (url.pathname.startsWith("/api/")) {
    event.respondWith(
      fetch(req)
        .then((res) => res)
        .catch(() =>
          new Response(JSON.stringify({ ok: false, offline: true }), {
            headers: { "Content-Type": "application/json" },
          })
        )
    );
    return;
  }

  // Shell: stale-while-revalidate
  event.respondWith(
    caches.match(req).then((cached) => {
      const fetched = fetch(req)
        .then((res) => {
          if (res && res.ok && url.origin === self.location.origin) {
            const clone = res.clone();
            caches.open(CACHE).then((c) => c.put(req, clone));
          }
          return res;
        })
        .catch(() => cached);
      return cached || fetched;
    })
  );
});
