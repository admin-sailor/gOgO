const CACHE_NAME = 'btts-cache-v1';
const IMG_EXT = ['.png', '.jpg', '.jpeg', '.svg', '.webp'];

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE_NAME));
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.map(k => (k !== CACHE_NAME ? caches.delete(k) : Promise.resolve())))
    )
  );
  self.clients.claim();
});

function isImageRequest(request) {
  const url = new URL(request.url);
  const extMatch = IMG_EXT.some(ext => url.pathname.toLowerCase().endsWith(ext));
  return request.destination === 'image' || extMatch;
}

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;
  if (!isImageRequest(request)) return;
  event.respondWith(
    caches.match(request).then(cached => {
      const fetchPromise = fetch(request)
        .then(response => {
          try {
            const clone = response.clone();
            caches.open(CACHE_NAME).then(cache => cache.put(request, clone));
          } catch (_) {}
          return response;
        })
        .catch(() => cached);
      return cached || fetchPromise;
    })
  );
});
