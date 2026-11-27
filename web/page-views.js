// Page-view counting for every published page (the demo, results/ and docs/),
// loaded once by each page so the counter's settings live in one place
// (eval/tests/test_site.py checks every page loads it). It counts page views
// only, with GoatCounter (https://www.goatcounter.com/help/js): no events, and
// nothing about the demo's camera, files, encodes or settings.
//
// The recorded path and referrer drop the query string and fragment: the demo's
// and results pages' URLs carry deep-link state (deep-link.js), and GoatCounter
// would otherwise record `pathname + search`. It also appends the counting
// notice to the page's footer, so a page that counts always says so.

const COUNT_ENDPOINT = 'https://nanocodec.goatcounter.com/count';
const COUNT_SCRIPT = 'https://gc.zgo.at/count.js';
const NOTICE = 'Page views are counted with GoatCounter (no cookies, no personal data).';
// Local development and the headless page tests serve from these hosts: the
// counter would skip them anyway, so its script is not fetched there at all
// (the tests then need no network).
const LOCAL_HOST = /^(localhost|127\.|\[::1\]$)/;

const withoutQuery = (url) => url.split(/[?#]/, 1)[0];

function appendNotice() {
  const footer = document.querySelector('body > footer');
  if (!footer) return;
  const note = document.createElement(footer.firstElementChild?.tagName === 'SPAN' ? 'span' : 'p');
  note.className = 'page-views-note';
  note.textContent = NOTICE;
  footer.append(note);
}

function loadCounter() {
  if (location.protocol === 'file:' || LOCAL_HOST.test(location.hostname)) return;
  window.goatcounter = { no_events: true, path: () => location.pathname, referrer: withoutQuery };
  const script = document.createElement('script');
  script.async = true;
  script.dataset.goatcounter = COUNT_ENDPOINT;
  script.src = COUNT_SCRIPT;
  document.head.append(script);
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', appendNotice);
else appendNotice();
loadCounter();
