// Charlie - Equity Analyzer Service Worker
// BUILD_VERSION is updated on each deploy to trigger cache invalidation
const BUILD_VERSION = '20260424-23';
const CACHE_NAME = 'charlie-' + BUILD_VERSION;

const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/icon-152.png',
  '/icon-167.png',
  '/icon-180.png',
  '/icon-192.png',
  '/icon-512.png'
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
  console.log('Service Worker: Installing', BUILD_VERSION);
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('Service Worker: Caching static assets');
        return cache.addAll(STATIC_ASSETS);
      })
      .then(() => self.skipWaiting())
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  console.log('Service Worker: Activating', BUILD_VERSION);
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            console.log('Service Worker: Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Web push — show native notification for payloads the backend sends.
// Payload shape (from media_trackers/notifications._push_send):
//   { title, body, url }
self.addEventListener('push', (event) => {
  let data = { title: 'Charlie', body: 'You have a new alert', url: '/' };
  try {
    if (event.data) {
      const raw = event.data.text();
      try { data = { ...data, ...JSON.parse(raw) }; }
      catch { data.body = raw; }
    }
  } catch (e) { console.warn('push parse error', e); }
  event.waitUntil(
    self.registration.showNotification(data.title, {
      body: data.body || '',
      icon: '/icon-192.png',
      badge: '/icon-152.png',
      data: { url: data.url || '/' },
      tag: data.url || 'charlie-alert',
      renotify: true,
    })
  );
});

// Clicking the notification focuses the existing Charlie tab if open, or
// opens a new one at the payload's url.
self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  const targetUrl = (event.notification.data && event.notification.data.url) || '/';
  event.waitUntil((async () => {
    const all = await self.clients.matchAll({ type: 'window', includeUncontrolled: true });
    for (const client of all) {
      if ('focus' in client) {
        try {
          if (client.url.includes(self.location.origin)) {
            await client.navigate(targetUrl);
            return client.focus();
          }
        } catch {}
      }
    }
    if (self.clients.openWindow) return self.clients.openWindow(targetUrl);
  })());
});

// Fetch event - network first, fallback to cache
self.addEventListener('fetch', (event) => {
  // Skip non-GET requests
  if (event.request.method !== 'GET') return;

  // Skip API calls, version checks, and external requests
  if (event.request.url.includes('/api/') ||
      event.request.url.includes('/version') ||
      !event.request.url.startsWith(self.location.origin)) {
    return;
  }

  event.respondWith(
    fetch(event.request)
      .then((response) => {
        // Clone the response before caching
        const responseClone = response.clone();
        caches.open(CACHE_NAME).then((cache) => {
          cache.put(event.request, responseClone);
        });
        return response;
      })
      .catch(() => {
        // Fallback to cache if network fails
        return caches.match(event.request);
      })
  );
});
