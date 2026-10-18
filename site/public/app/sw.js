// Offline cache for the TRACE app. Bump VERSION whenever data.json changes.
const VERSION = 'trace-v2';
const ASSETS = ['./', './index.html', './data.json', './manifest.webmanifest', './icon-192.png', './icon-512.png', '../favicon.png'];
self.addEventListener('install', e => e.waitUntil(caches.open(VERSION).then(c => c.addAll(ASSETS)).then(() => self.skipWaiting())));
self.addEventListener('activate', e => e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== VERSION).map(k => caches.delete(k)))).then(() => self.clients.claim())));
self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  // network-first so a redeploy is picked up; fall back to cache offline
  e.respondWith(fetch(e.request).then(r => { const c = r.clone(); caches.open(VERSION).then(x => x.put(e.request, c)); return r; })
    .catch(() => caches.match(e.request)));
});
