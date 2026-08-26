/**
 * Tests for the frontend markdown renderer's escaping contract.
 *
 * renderMarkdown() output goes straight into dangerouslySetInnerHTML, and its
 * input is model output or document text -- neither is trusted markup. The two
 * properties that matter pull in opposite directions, so both are pinned here:
 *
 *   1. Untrusted INPUT must be escaped, or a research PDF containing markup
 *      gets executed, and ordinary prose like "EPS > $2" renders broken.
 *   2. The renderer's OWN output must NOT be escaped, or the page fills with
 *      visible tags. An earlier fix escaped both and produced exactly that.
 *
 * The function is extracted from src/app.jsx by marker rather than copied, so
 * this cannot silently drift from what ships.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const src = readFileSync(join(root, 'src', 'app.jsx'), 'utf-8');

const START = 'const escapeHtml = (str)';
const END = 'const renderMarkdown';
const startIdx = src.indexOf(START);
if (startIdx < 0) throw new Error('escapeHtml helper not found in src/app.jsx');

// Take from the helpers through the end of renderMarkdown, located by its
// distinctive closing rather than a line number.
const CLOSE = '            return html;\n        };';
const closeIdx = src.indexOf(CLOSE, src.indexOf(END, startIdx));
if (closeIdx < 0) throw new Error('renderMarkdown close marker not found in src/app.jsx');

const source = src.slice(startIdx, closeIdx + CLOSE.length);

const preamble = `const safeStr = (v, f='') => (v==null?f:String(v));\n`;
// DOMPurify is loaded from a CDN in index.html. Testing with it ABSENT is the
// interesting case: that is when the fallbacks decide whether the app is safe.
const make = () => {
  const fn = new Function('DOMPurify', `${preamble}${source}\nreturn {renderMarkdown, sanitizeHtml, escapeHtml};`);
  return fn(undefined);
};

const { renderMarkdown, sanitizeHtml } = make();

let failed = 0;
const check = (name, cond, got) => {
  console.log(`${cond ? 'ok   ' : 'FAIL '} ${name}`);
  if (!cond) { failed++; if (got !== undefined) console.log(`       got: ${String(got).slice(0, 200)}`); }
};

// --- input is escaped ------------------------------------------------------
let out = renderMarkdown('Hello <script>alert(1)</script>');
check('script tag is escaped, not executed',
      !out.includes('<script>') && out.includes('&lt;script&gt;'), out);

out = renderMarkdown('<img src=x onerror="alert(1)">');
check('img onerror payload is escaped', !out.includes('<img ') && out.includes('&lt;img'), out);

out = renderMarkdown('EPS > $2 and margin <5% this year');
check('ordinary financial prose with > and < survives',
      out.includes('&gt;') && out.includes('&lt;5%'), out);

// --- markdown still works --------------------------------------------------
check('bold renders', renderMarkdown('Revenue is **up sharply** now').includes('<strong>up sharply</strong>'));
check('italic renders', renderMarkdown('This is *emphasised* text').includes('<em>emphasised</em>'));

out = renderMarkdown('| Metric | Value |\n| --- | --- |\n| EPS | 2.10 |');
check('tables render', out.includes('<table') && out.includes('EPS'), out);

out = renderMarkdown('Margin **> 45%** achieved');
check('bold wrapping an escaped comparison', out.includes('<strong>&gt; 45%</strong>'), out);

// --- the renderer's own output must stay real HTML -------------------------
out = renderMarkdown('Plain sentence.');
check('output is HTML, not escaped gibberish, when DOMPurify is missing',
      out.includes('<p') && !out.includes('&lt;p'), out);

// --- model-authored HTML fails closed --------------------------------------
out = sanitizeHtml('<p>ok</p><script>alert(1)</script>');
check('sanitizeHtml escapes rather than injecting raw HTML when DOMPurify is missing',
      !out.includes('<script>'), out);

console.log(failed ? `\n${failed} failed` : '\nall passed');
process.exit(failed ? 1 : 0);
