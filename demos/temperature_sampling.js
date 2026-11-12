// Temperature and sampling demo. Data: TEMP_DATA (make_L16_demo_temperature.py). All distributions computed here.
(function () {
  const D = window.TEMP_DATA;
  if (new URLSearchParams(location.search).get('embed') === '1') document.body.classList.add('embed');
  const $ = id => document.getElementById(id);
  const TS = [0, 0.5, 0.7, 1.0, 1.5], NSHOW = 15;
  let pi = 0, seed = 1, rng = null, draws = [], lastHit = -1;

  function mulberry(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function vis(s) { return s.replace(/ /g, '·').replace(/\n/g, '↵'); }
  const pct = x => x >= 0.1 ? (100 * x).toFixed(1) + '%' : x >= 0.001 ? (100 * x).toFixed(2) + '%' : x === 0 ? '0%' : (100 * x).toFixed(3) + '%';
  const fmt = n => n >= 100 ? Math.round(n).toLocaleString('en-US') : n >= 10 ? n.toFixed(1) : n.toFixed(2);

  // Distribution actually sampled from: softmax(logits / T), then top-p cut and renormalization.
  function dist(T, topP) {
    const l = D.prompts[pi].logits, n = l.length, p = new Float64Array(n);
    let s = 0;
    for (let i = 0; i < n; i++) { p[i] = Math.exp(l[i] / T); s += p[i]; }
    let k = n, c = 0;
    for (let i = 0; i < n; i++) { p[i] /= s; if (topP < 1) { if (c >= topP) { k = i; break; } c += p[i]; } }
    if (k < n) { let z = 0; for (let i = 0; i < k; i++) z += p[i]; for (let i = 0; i < k; i++) p[i] /= z; for (let i = k; i < n; i++) p[i] = 0; }
    let H = 0; for (let i = 0; i < k; i++) if (p[i] > 0) H -= p[i] * Math.log(p[i]);
    return { p, k, eff: Math.exp(H) };
  }
  function name(i) { const pc = D.prompts[pi].pieces; return i < pc.length ? vis(pc[i]) : 'rank ' + (i + 1).toLocaleString('en-US'); }

  function render() {
    const T = +$('temp').value, tp = +$('topp').value;
    $('temp-out').textContent = T.toFixed(2); $('topp-out').textContent = tp.toFixed(2);
    const { p, k, eff } = dist(T, tp);
    $('eff').textContent = fmt(eff);
    let top = 0, shown = 0;
    for (let i = 0; i < NSHOW; i++) { top = Math.max(top, p[i]); shown += p[i]; }
    const other = Math.max(0, 1 - shown);
    const scale = Math.min(1, Math.max(0.02, Math.ceil(top * 1.05 * 50) / 50)); // nice axis max for the 15 pieces
    $('scale-note').textContent = 'Bar length: full width = ' + (100 * scale).toFixed(0) + '% (the grey bar may run off the end).';
    const rows = [];
    for (let i = 0; i < NSHOW; i++) {
      const cutp = p[i] === 0, w = Math.min(100, 100 * p[i] / scale);
      rows.push(`<div class="lab">${name(i)}</div><div class="track"><div class="bar${cutp ? ' cut' : ''}${i === lastHit ? ' drawn-hit' : ''}" style="width:${w}%"></div></div><div class="pct${cutp ? ' cutlab' : ''}">${cutp ? 'cut' : pct(p[i])}</div>`);
    }
    const nOther = p.length - NSHOW;
    rows.push(`<div class="lab" style="font-family:system-ui,sans-serif;font-size:16px">all other</div><div class="track"><div class="bar other${lastHit >= NSHOW ? ' drawn-hit' : ''}" style="width:${Math.min(100, 100 * other / scale)}%"></div></div><div class="pct">${other < 0.0005 && other > 0 ? '<0.1%' : pct(other)}${other > scale ? ' →' : ''}</div>`);
    $('bars').innerHTML = rows.join('');
    const top5 = Array.from({ length: 5 }, (_, i) => name(i) + ' ' + pct(p[i])).join(', ');
    $('summary').textContent = `${D.prompts[pi].prompt} ... Temperature ${T.toFixed(2)}, top-p ${tp.toFixed(2)}: ${fmt(eff)} effective choices (e to the entropy), ${k < p.length ? k.toLocaleString('en-US') : 'all ' + p.length.toLocaleString('en-US')} pieces kept. Top 5 pieces: ${top5}. All ${nOther.toLocaleString('en-US')} other pieces together: ${pct(other)}.`;
    // continuation at nearest grid point
    const gT = TS.reduce((a, b) => Math.abs(b - T) < Math.abs(a - T) ? b : a), gP = tp < 0.95 ? 0.9 : 1.0;
    const gTs = Number.isInteger(gT) ? gT.toFixed(1) : String(gT), key = gT === 0 ? '0|1.0' : gTs + '|' + gP.toFixed(1);
    const pieces = D.prompts[pi].cont[key];
    $('cont-label').textContent = gT === 0 ? 'Greedy (T = 0): always the most likely piece.' : `T = ${gTs}, top-p ${gP.toFixed(1)}${(gT === T && gP === tp) ? '' : ' (nearest to your settings)'}`;
    const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');
    $('cont-text').innerHTML = `<span class="prompt">${esc(D.prompts[pi].prompt)}</span><span class="gen">${pieces.map(s => esc(s).replace(/\n/g, '<span class="nl">↵</span>\n')).join('')}</span>`;
  }

  function draw() {
    const { p } = dist(+$('temp').value, +$('topp').value);
    const u = rng(); let c = 0, i = 0;
    for (; i < p.length - 1; i++) { c += p[i]; if (u < c) break; }
    lastHit = i; draws.push(name(i));
    $('drawn').innerHTML = `Draw ${draws.length} (seed ${seed}): <b>${name(i)}</b>, probability ${pct(p[i])}, rank ${(i + 1).toLocaleString('en-US')}.<br><span class="k">So far: ${draws.slice(-8).join(' ')}</span>`;
    render();
  }
  function reseed(first) { if (!first) seed++; rng = mulberry(seed * 7919); draws = []; lastHit = -1; $('drawn').textContent = 'Seed ' + seed + ': draws are reproducible.'; }

  D.prompts.forEach((pr, i) => {
    const b = document.createElement('button'); b.textContent = pr.prompt + ' …'; b.setAttribute('aria-pressed', i === 0);
    b.onclick = () => { pi = i; [...$('prompts').children].forEach((x, j) => x.setAttribute('aria-pressed', j === i)); reseed(true); render(); };
    $('prompts').appendChild(b);
  });
  $('temp').oninput = $('topp').oninput = () => { lastHit = -1; render(); };
  $('draw').onclick = draw;
  $('seed').onclick = () => { reseed(false); render(); };
  reseed(true); render();
})();
