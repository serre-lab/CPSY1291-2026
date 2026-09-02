// check_overflow_bootcamp.js — the v8 overflow check, pointed at the recitation decks.
//
// The theme locks `section { height: 720px; overflow: hidden }`, so any slide
// taller than that is silently cropped. slides/check_overflow.js hard-codes the
// v8 lecture directory; this is the same measurement for 2026/bootcamp/.
//
//   node build/check_overflow_bootcamp.js          # all decks
//   node build/check_overflow_bootcamp.js 03       # only decks matching "03"
//
// Requires Google Chrome (set CHROME_PATH to override).
const fs = require('fs');
const path = require('path');

const BOOTCAMP = path.join(__dirname, '..');
const SLIDES = process.env.CPSY_SLIDES_DIR || path.join(BOOTCAMP, '..', '..', 'slides');

const puppeteer = require(path.join(SLIDES, 'node_modules/puppeteer-core'));
const engineFactory = require(path.join(SLIDES, 'engine.js'));

const CHROME = process.env.CHROME_PATH ||
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const themeCSS = fs.readFileSync(path.join(SLIDES, 'template/cpsy1291.css'), 'utf8');
const TMP = path.join(BOOTCAMP, '_ovf_check.html');
const THRESHOLD = 4;

(async () => {
  const filter = process.argv[2] || '';
  const decks = fs.readdirSync(BOOTCAMP)
    .filter(f => /^recitation-\d+.*\.md$/.test(f) && f.includes(filter)).sort();

  const browser = await puppeteer.launch({
    executablePath: CHROME, headless: 'new', args: ['--no-sandbox'],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 900 });

  const report = [];
  for (const deck of decks) {
    const md = fs.readFileSync(path.join(BOOTCAMP, deck), 'utf8');
    const marp = engineFactory({ html: true });
    marp.themeSet.add(themeCSS);
    const { html, css } = marp.render(md);
    fs.writeFileSync(TMP,
      `<!DOCTYPE html><html><head><meta charset="utf8"><style>${css}</style></head><body>${html}</body></html>`);
    await page.goto('file://' + TMP, { waitUntil: 'networkidle0' });
    try { await page.evaluate(() => document.fonts.ready); } catch (e) {}

    const flagged = await page.evaluate((thr) => {
      const out = [];
      document.querySelectorAll('section').forEach((s, i) => {
        const over = s.scrollHeight - s.clientHeight;
        if (over > thr) {
          const h = s.querySelector('h1,h2,h3');
          out.push({ slide: i + 1, over: Math.round(over),
                     title: h ? h.textContent.trim().slice(0, 55) : '(no heading)' });
        }
      });
      return { total: document.querySelectorAll('section').length, out };
    }, THRESHOLD);

    report.push({ deck, slides: flagged.total, flagged: flagged.out });
  }
  await browser.close();
  if (fs.existsSync(TMP)) fs.unlinkSync(TMP);

  let nClipped = 0;
  console.log('=== bootcamp overflow report ===');
  for (const r of report) {
    if (r.flagged.length === 0) {
      console.log(`  OK   ${r.deck}  (${r.slides} slides)`);
    } else {
      nClipped += r.flagged.length;
      console.log(`  WARN ${r.deck}  (${r.slides} slides) — ${r.flagged.length} clipped:`);
      for (const s of r.flagged)
        console.log(`        slide ${s.slide}: +${s.over}px over  "${s.title}"`);
    }
  }
  console.log(`\n=== ${nClipped} clipped slide(s) across ${report.length} decks ===`);
})().catch(e => { console.error(e); process.exit(1); });
