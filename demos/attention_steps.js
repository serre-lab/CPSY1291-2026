/* Progressive presentation of one head; the overview is explicitly schematic. */
(() => {
  'use strict';
  const C = window.AttentionExample;
  const $ = id => document.getElementById(id);
  const colors = { ink:'#1E293B', q:'#B5396B', k:'#3E6D8E', v:'#3F7A6B', muted:'#77736b', line:'#dad7ce', paper:'#FAF9F5' };
  const params = new URLSearchParams(location.search);
  if (params.get('embed') === '1') document.body.classList.add('embed');
  const state = { view:params.get('view') === 'transformer' ? 'transformer' : 'attention', step:0, attentionStep:0, transformerStep:0, query:2, mask:true, scale:true, equal:false, zeroValue:false, playing:false };
  const attentionLabels = ['Input', 'Q, K, V', 'Match', 'Scale & mask', 'Softmax', 'Gather', 'All positions'];
  const transformerLabels = ['Embed', 'Attention', 'Add back', 'Small network', 'Repeat', 'Predict'];
  const attentionTitles = ['A word becomes a vector at a position', 'One input, three learned projections', 'Compare one query with every key', 'Scale the scores; exclude the future', 'Turn scores into shares of attention', 'Bring the values back, in proportion', 'Every position does the same operation'];
  const transformerTitles = ['Three positions enter three parallel streams', 'Attention lets positions exchange information', 'Add the attention result to the original stream', 'A small network works on each position', 'Add again, then repeat the block', 'Read the final position to predict what comes next'];
  let timer = null;
  const f = (n, digits=2) => n === -Infinity ? '−∞' : n.toFixed(digits).replace(/\.0+$|(?<=\.[0-9]*?)0+$/g, '').replace(/^-/, '−');
  const vec = (row, digits=2) => `[${row.map(v => f(v,digits)).join(', ')}]`;
  const txt = (x,y,value,cls='',anchor='middle') => `<text x="${x}" y="${y}" text-anchor="${anchor}" class="${cls}">${value}</text>`;
  const rect = (x,y,w,h,fill='none',stroke=colors.line,r=10,extra='') => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}" ${extra}/>`;
  const W = 2.5; // one stroke width for every stream / flow arrow; arrowheads are fixed-size (userSpaceOnUse), tip exactly at the path end
  const path = (d,color=colors.line,width=W,arrow=false,extra='') => `<path d="${d}" fill="none" stroke="${color}" stroke-width="${width}" ${arrow ? `marker-end="url(#${arrow==='scaled'?'arrowW':'arrow'}-${Object.keys(colors).find(k=>colors[k]===color)||'ink'})"` : ''} ${extra}/>`;
  const line = (x,y,x2,y2,color=colors.line,width=W,arrow=false) => path(`M${x},${y} L${x2},${y2}`,color,width,arrow);
  function packet(d,color=colors.v,delay=0,r=5) {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return '';
    return `<circle r="${r}" fill="${color}" opacity="0"><animateMotion path="${d}" dur="1.7s" begin="${delay}s" fill="freeze"/><animate attributeName="opacity" values="0;1;1;0" keyTimes="0;0.08;0.88;1" dur="1.7s" begin="${delay}s" fill="freeze"/></circle>`;
  }
  const xs = [173,520,867];
  const defs = () => `<defs>${Object.entries(colors).map(([k,c])=>`<marker id="arrow-${k}" viewBox="0 0 10 9" refX="10" refY="4.5" markerWidth="10" markerHeight="9" markerUnits="userSpaceOnUse" orient="auto"><path d="M0,0 L10,4.5 L0,9 Z" fill="${c}" stroke="none"/></marker><marker id="arrowW-${k}" viewBox="0 0 3.4 3" refX="3.4" refY="1.5" markerWidth="3.4" markerHeight="3" markerUnits="strokeWidth" orient="auto"><path d="M0,0 L3.4,1.5 L0,3 Z" fill="${c}" stroke="none"/></marker>`).join('')}<pattern id="masked" patternUnits="userSpaceOnUse" width="8" height="8"><path d="M0,8 L8,0" stroke="#dad7ce" stroke-width="1"/></pattern></defs>`;
  const stepCount = () => state.view === 'attention' ? 7 : 6;
  function stop() { clearInterval(timer); timer = null; state.playing = false; $('play').textContent = '▶ Play'; }
  function go(step) { stop(); state.step = Math.max(0,Math.min(stepCount()-1,step)); render(); }
  function setView(view) {
    stop(); state[state.view+'Step'] = state.step;
    state.view = view; state.step = state[view+'Step']; render();
  }
  function setQuery(index) { stop(); state.query = index; render(); }
  function renderInput(r) {
    let s = '';
    xs.forEach((x,j) => {
      s += txt(x,25,'word embedding','small');
      s += rect(x-130,40,260,52,colors.paper) + txt(x,74,vec(C.E[j]),'math');
      s += txt(x-148,151,'+','large') + txt(x,121,'position embedding','small');
      s += rect(x-130,136,260,52,colors.paper) + txt(x,170,vec(C.P[j]),'math');
      const d = `M${x},188 L${x},257`;
      s += path(d,colors.ink,W,true) + packet(d,colors.ink,j*.16);
      s += rect(x-130,257,260,59,'#eeede6',colors.ink) + txt(x,294,vec(r.X[j]),'math');
      s += txt(x,342,`x${j+1} = embedding + position`,'small');
    });
    return s;
  }
  function renderProjections(r) {
    let s='';
    xs.forEach((x,j) => {
      s += txt(x,29,`x${j+1} = ${vec(r.X[j])}`,'math');
      const triples = [['q',-106,'W<tspan baseline-shift="sub" font-size="72%">Q</tspan>',r.Q[j]],['k',0,'W<tspan baseline-shift="sub" font-size="72%">K</tspan>',r.K[j]],['v',106,'W<tspan baseline-shift="sub" font-size="72%">V</tspan>',r.V[j]]];
      triples.forEach(([name,dx,w,v],i) => {
        const d=`M${x},47 C${x},92 ${x+dx},90 ${x+dx},137`;
        s += path(d,colors[name],W,true) + packet(d,colors[name],i*.15);
        s += rect(x+dx-45,137,90,42,colors.paper,colors[name]) + txt(x+dx,165,w,`${name}text label vector`);
        s += line(x+dx,179,x+dx,230,colors[name],W,true);
        s += txt(x+dx,254,name,`${name}text large vector`);
        s += txt(x+dx,285,vec(v),`${name}text math`);
      });
    });
    return s + txt(520,341,'The same three matrices are applied at every position.','small');
  }
  function renderMatch(r,masked) {
    const i=state.query, q=r.Q[i]; let s='';
    xs.forEach((x,j)=>{
      const blocked=masked && state.mask && j>i;
      const color=blocked ? colors.muted : colors.k;
      s += `<g ${blocked?'class="faded"':''}>`;
      s += txt(x,33,`k = ${vec(r.K[j])}`,'ktext math');
      s += rect(x-140,53,280,83,colors.paper,colors.line);
      s += txt(x,84,`${f(q[0])} × ${f(r.K[j][0])} + ${f(q[1])} × ${f(r.K[j][1])}`,'math');
      s += txt(x,119,`score = ${f(r.scores[i][j])}`,'large');
      s += '</g>';
      if(masked) {
        s += txt(x,167,state.scale ? `${f(r.scores[i][j])} ÷ √2 ≈ ${f(r.scaled[i][j])}` : `no scaling: ${f(r.scaled[i][j])}`,'math');
        s += rect(x-140,187,280,51,blocked?'url(#masked)':'#3e6d8e0c',blocked?colors.line:colors.k);
        s += txt(x,220,blocked?'−∞  ·  future':state.equal?'0  ·  equal-weight override':f(r.logits[i][j]),blocked?'label':'ktext label');
      }
      const yEnd=masked?238:136, d = `M520,305 C520,270 ${x},${yEnd+40} ${x},${yEnd}`;
      s += path(d,color,blocked?1:W,true,blocked?'stroke-dasharray="4 5"':'');
      if(!blocked) s += packet(d,colors.q,j*.18);
    });
    s += rect(361,305,318,49,'#b5396b08',colors.q) + txt(520,337,`q(${C.words[i]}) = ${vec(q)}`,'qtext math');
    return s;
  }
  function renderSoftmax(r) {
    const i=state.query; let s='';
    xs.forEach((x,j)=>{
      const blocked=state.mask && j>i;
      s += txt(x,28,`score = ${f(r.logits[i][j])}`,'math');
      s += line(x,44,x,72,colors.ink,W,true);
      s += txt(x,106,blocked?'exp(−∞) = 0':`exp(${f(r.logits[i][j])} − ${f(r.maxima[i])})`,'math');
      s += txt(x,143,`${r.exponents[i][j]===0||r.exponents[i][j]===1?'=':'≈'} ${f(r.exponents[i][j],3)}`,'math');
      s += line(x,160,x,190,colors.ink,W,true);
      s += txt(x,220,`${f(r.exponents[i][j],3)} ÷ ${f(r.sums[i],3)}`,'math');
      s += rect(x-132,240,264,38,colors.paper,colors.line,6);
      if(r.A[i][j]>0) s += rect(x-132,240,264*r.A[i][j],38,colors.k,'none',6,'class="bar-grow"');
      s += txt(x,314,`${(100*r.A[i][j]).toFixed(1)}%`,'ktext large');
    });
    return s + txt(520,351,'Divide each exponential by their shared sum → weights sum to 1.','small');
  }
  function renderGather(r) {
    const i=state.query;let s='';
    xs.forEach((x,j)=>{
      const a=r.A[i][j], contribution=r.V[j].map(v=>v*a);
      s += `<g ${a===0?'class="faded"':''}>`;
      s += txt(x,31,`v = ${vec(r.V[j])}`,'vtext math');
      s += txt(x,74,`× ${f(a,3)}`,'ktext large');
      s += line(x,88,x,136,colors.v,W,true);
      s += rect(x-142,136,284,57,'#3f7a6b08',colors.v) + txt(x,172,vec(contribution,3),'vtext math');
      s += '</g>';
      // each value lands at its own point on the result box, so the arrowheads do not pile up
      const ex=410+110*j;
      const d=`M${x},193 C${x},246 ${ex},247 ${ex},300`;
      s += path(d,a===0?colors.line:colors.v,a===0?1:W+6*a,'scaled',a===0?'stroke-dasharray="4 5"':'');
      if(a>0) s += packet(d,colors.v,j*.3,4+5*a);
    });
    s += rect(340,300,360,56,'#b5396b08',colors.q) + txt(520,336,`z(${C.words[i]}) ≈ ${vec(r.Z[i],3)}`,'qtext math');
    return s;
  }
  function renderAll(r) {
    let s=txt(305,24,'A: how much each position reads','label') + txt(801,24,'Z = A V: head outputs','label');
    s += txt(305,56,'columns = keys / sources','small');
    C.words.forEach((w,j)=>s+=txt(217+j*117,87,w,'ktext label'));
    r.A.forEach((row,i)=>{
      const y=105+i*77;
      s+=`<g class="matrix-row" data-row="${i}" tabindex="0" role="button" aria-label="Inspect the ${C.words[i]} query row">`;
      s+=rect(48,y-4,490,71,'none',state.query===i?colors.q:'none',8,'class="row-focus"');
      s+=txt(133,y+36,C.words[i],state.query===i?'qtext label':'label','end');
      row.forEach((w,j)=>{
        s+=rect(164+j*117,y,107,63,w===0?'url(#masked)':colors.k,'none',5,w>0?`fill-opacity="${.1+.78*w}"`:'');
        s+=`<text x="${217+j*117}" y="${y+38}" text-anchor="middle" class="math" style="fill:${w>.6?colors.paper:colors.ink}">${f(w,3)}</text>`;
      });
      s+=txt(801,y+37,vec(r.Z[i],3),'vtext math');
      s+='</g>';
    });
    s+=txt(65,352,'rows = queries / receivers','small','start');
    s+=txt(607,217,'× V =','large');
    s+=rect(686,105,234,217,'none',colors.v,6);
    return s;
  }
  function explainAttention(r) {
    const i=state.query, word=C.words[i], a=r.A[i];
    const base = [
      `<h2>The same word has a position.</h2><p>Add a word embedding to a position embedding. Each resulting row <b>x</b> has three coordinates.</p><div class="formula">x(${word})<br>= ${vec(C.E[i])}<br>+ ${vec(C.P[i])}<br>= <b>${vec(r.X[i])}</b></div><p class="subtle">These tiny embeddings and weights are hand-set for arithmetic, not trained language representations.</p>`,
      `<h2>Three roles, one source.</h2><p><b class="q">Query:</b> what to match.<br><b class="k">Key:</b> what can be matched.<br><b class="v">Value:</b> content to pass on.</p><div class="formula">q = x W<sub>Q</sub><br>k = x W<sub>K</sub><br>v = x W<sub>V</sub></div><p class="subtle">Training learns the matrices. A new input produces new queries, keys and values.</p>`,
      `<h2>One query, three comparisons.</h2><p>The query of <b>${word}</b> takes a dot product with every key. Multiply matching coordinates, then add.</p><div class="formula">q(${word}) · k(cat)<br>= ${f(r.Q[i][0])}×2 + ${f(r.Q[i][1])}×0<br>= <b>${f(r.scores[i][1])}</b></div><p class="subtle">A score is a match, not yet a probability. Each word has its own query.</p>`,
      `<h2>Prepare the scores.</h2><p>${state.scale?'Divide by √2 because each query and key has two coordinates.':'Scaling is off: larger score differences make softmax sharper.'}</p><p>${state.mask?`The query at <b>${word}</b> can read itself and earlier positions. Future scores become −∞.`:'The mask is off: all three positions can read one another.'}</p><p class="subtle">Select <b>cat</b> above to make the future exclusion visible.</p>`,
      `<h2>Softmax makes a weighted choice.</h2><div class="formula">a<sub>j</sub> = exp(s<sub>j</sub> − m)<br>     / Σ exp(s − m)<br>m = max(s) = ${f(r.maxima[i])}</div><p class="subtle">Subtracting the maximum keeps the exponentials small and leaves the probabilities unchanged. A masked entry has exp(−∞) = 0.</p>`,
      `<h2>Read content, not keys.</h2><p>Multiply each <b class="v">value</b> by its attention weight, then add the contributions.</p><div class="formula">z = Σ a<sub>j</sub> v<sub>j</sub><br>≈ <b>${vec(r.Z[i],3)}</b></div><p class="insight">${state.zeroValue?'Cat’s attention weight is unchanged. Its contribution is now zero.':'Zero cat’s value: will the weights change, the result change, or both?'}</p>`,
      `<h2>One row per receiving position.</h2><p>Each row is one query’s weights. Multiply by V to get all three outputs.</p><div class="formula">A = softmax(QKᵀ/√2 + M)<br>Z = A V</div><p class="subtle">M masks future scores with −∞. Select a row.</p><button class="jump" id="zoom-out">Where does Z go? ↗</button>`
    ];
    let html=base[state.step];
    if(state.step===6 && (!state.scale || !state.mask || state.equal)) {
      html=html.replace('A = softmax(QKᵀ/√2 + M)',state.equal?'Aᵢⱼ = 1/nᵢ (allowed)':`A = softmax(QKᵀ${state.scale?'/√2':''}${state.mask?' + M':''})`);
      if(state.equal)html=html.replace('M masks future scores with −∞. Select a row.','nᵢ counts allowed sources. Select a row.');
    }
    return html;
  }
  function architectureMap(step) {
    const labels=['Embedding + position','LN → attention → Wₒ','+ residual','LN → small network','+ residual · repeat','Final LN → readout'];
    let s=txt(140,16,'Decoder transformer','small');
    s+=rect(15,77,250,211,'none',colors.line,12,'stroke-dasharray="5 5"');
    labels.forEach((label,j)=>{
      const y=29+j*51,active=j===step;
      if(j>0)s+=line(140,y-15,140,y,active?colors.q:colors.line,W,true);
      s+=rect(26,y,228,36,active?'#b5396b10':colors.paper,active?colors.q:colors.line,7);
      s+=txt(140,y+24,label,active?'qtext tiny':'tiny');
    });
    s+=txt(140,354,'LN = layer normalization','tiny');
    return s;
  }
  const lanes=[430,665,900];
  function renderTransformer(r) {
    const step=state.step;let s=architectureMap(step);
    s+=line(293,7,293,352,colors.line,1);
    if(step===0) {
      lanes.forEach((x,j)=>{
        s+=txt(x,41,C.words[j],'large');
        s+=line(x,58,x,116,colors.ink,W,true);
        s+=rect(x-91,116,182,69,colors.paper,colors.k)+txt(x,145,'word + position','label')+txt(x,171,'embeddings','small');
        const d=`M${x},185 L${x},294`;
        s+=path(d,colors.ink,W,true)+packet(d,colors.k,j*.18);
        s+=rect(x-83,294,166,51,'#3e6d8e0a',colors.k)+txt(x,327,`x${j+1}`,'large vector');
      });
    } else if(step===1) {
      lanes.forEach((x,j)=>{
        s+=txt(x,31,C.words[j],'large')+line(x,45,x,84,colors.ink,W,true);
        s+=rect(x-72,84,144,40,colors.paper,colors.line)+txt(x,111,'Layer norm','label');
        s+=txt(x,163,'q, k, v','label');
      });
      for(let receiver=0;receiver<3;receiver++)for(let source=0;source<=receiver;source++) {
        // spread the starts (by receiver) and the ends (by source) so no two arrows share a point
        const sx=lanes[source]+22*(receiver-source), ex=lanes[receiver]-50+50*source;
        const d=`M${sx},177 C${sx},228 ${ex},232 ${ex},287`;
        s+=path(d,colors.v,W,true)+packet(d,colors.v,receiver*.22+source*.1);
      }
      lanes.forEach((x,j)=>s+=rect(x-83,287,166,50,'#3f7a6b08',colors.v)+txt(x,319,`z${j+1}`,'vtext large vector'));
      s+=txt(665,358,'Only attention mixes information across positions.','small');
    } else if(step===2 || step===3 || step===4) {
      lanes.forEach((x,j)=>{
        const bypass=x-66, branch=x+28;
        s+=txt(x,30,step===2?`x${j+1}`:`h${j+1}`,'large vector');
        s+=line(x,42,x,62,colors.ink,W);
        const residual=`M${x},62 L${bypass},62 L${bypass},292 L${x-18},292`;
        s+=path(residual,colors.ink,W,true)+packet(residual,colors.ink,j*.12);
        s+=path(`M${x},62 L${branch},62 L${branch},${step===2?105:104}`,colors.ink,W,true);
        if(step===2) {
          s+=rect(branch-61,105,122,54,'#3f7a6b08',colors.v)+txt(branch,128,'attention','label')+txt(branch,150,'result z','small');
          s+=line(branch,159,branch,194,colors.v,W,true);
          s+=rect(branch-61,194,122,39,colors.paper,colors.v)+txt(branch,221,'Wₒ','vtext vector');
        } else {
          s+=rect(branch-63,104,126,38,colors.paper,colors.line)+txt(branch,130,'Layer norm','small');
          s+=line(branch,142,branch,180,colors.k,W,true);
          s+=rect(branch-63,180,126,57,'#3e6d8e08',colors.k)+txt(branch,204,'small','label')+txt(branch,227,'network','label');
        }
        const d=`M${branch},${step===2?233:237} L${branch},255 L${x},255 L${x},274`;
        s+=path(d,step===2?colors.v:colors.k,W,true)+packet(d,step===2?colors.v:colors.k,j*.16);
        s+=`<circle cx="${x}" cy="292" r="18" stroke="${colors.ink}" fill="${colors.paper}" stroke-width="2"/>`+txt(x,299,'+','large');
        s+=line(x,310,x,343,colors.ink,W,true);
      });
      s+=txt(665,368,step===2?'h = x + Attention(LN(x)) Wₒ':step===3?'Shared network weights; separate computation at each position.':'Repeat these two additions with new weights in each block.','small');
    } else {
      lanes.forEach((x,j)=>{
        s+=txt(x,26,`final h${j+1}`,'label');
        s+=line(x,40,x,84,j===2?colors.q:colors.line,W,true);
        s+=rect(x-89,84,178,46,colors.paper,j===2?colors.q:colors.line)+txt(x,114,'Final layer norm','small');
        if(j<2)s+=txt(x,171,'earlier position','small');
      });
      s+=line(900,130,900,180,colors.q,W,true);
      s+=rect(794,180,212,51,'#b5396b08',colors.q)+txt(900,212,'Vocabulary readout','label');
      s+=line(900,231,900,278,colors.q,W,true);
      s+=rect(794,278,212,45,colors.paper,colors.q)+txt(900,308,'Softmax','label');
      s+=txt(900,356,'on · down · …','qtext large');
      s+=txt(533,239,'one score per','label')+txt(533,267,'possible next token','label');
      s+=txt(533,320,'Schematic candidates;','small')+txt(533,344,'no trained probabilities shown.','small');
      s+=packet('M900,130 L900,180',colors.q)+packet('M900,231 L900,278',colors.q,.5);
    }
    return s;
  }
  function explainTransformer() {
    return [
      '<h2>Keep a stream for each position.</h2><p>Each word starts with a vector. Each block updates these parallel vectors.</p><p class="insight">The <b>residual stream</b> carries the current representation forward. Branches add information to it.</p><p class="subtle">A schematic decoder transformer. Attention is the focus here; the other components set the context.</p>',
      '<h2>This is the communication step.</h2><p>A normalized copy of the stream produces queries, keys and values. Each position gathers information from itself and earlier positions.</p><button class="jump" id="zoom-in">Zoom into attention ↗</button><p class="subtle">Multiple heads do this in parallel. The walkthrough follows one head.</p>',
      '<h2>An update, plus the original.</h2><p>The output projection <b>Wₒ</b> maps the attention result back to the stream’s width. Then add it to the input.</p><div class="formula">h = x + update</div><p class="subtle">With multiple heads, concatenate their results before Wₒ. The bypass preserves a direct path for the input.</p>',
      '<h2>Now transform each position.</h2><p>Normalize a copy, then apply a small feedforward network. It can combine features within a vector.</p><p class="insight">The same network acts at every position, but it does not read any other position.</p>',
      '<h2>Build richer representations.</h2><p>Add the network’s output back into the stream. Repeat attention and the small network in the next block.</p><p>Each block has its own learned weights and recomputes Q, K and V from its input.</p><p class="subtle">The connections are shown in teaching order. Positions within each layer compute in parallel.</p>',
      '<h2>A different softmax.</h2><p><b>Attention:</b> weights over positions to read.</p><p><b>Output:</b> probabilities over vocabulary tokens to predict.</p><p class="insight">During generation, use the last position. During training, every position predicts the token that follows it.</p>'
    ][state.step];
  }
  function render() {
    const focused=document.activeElement;
    const restoreFocus=focused?.dataset.step!==undefined?`[data-step="${focused.dataset.step}"]`:focused?.dataset.query!==undefined?`[data-query="${focused.dataset.query}"]`:focused?.dataset.row!==undefined?`[data-row="${focused.dataset.row}"]`:null;
    const r=C.compute(state), att=state.view==='attention';
    const labels=att?attentionLabels:transformerLabels;
    $('attention-view').setAttribute('aria-pressed',att);
    $('transformer-view').setAttribute('aria-pressed',!att);
    $('steps').innerHTML=labels.map((label,i)=>`<button data-step="${i}" ${i===state.step?'aria-current="step"':''}><span class="step-index">${i+1}</span>${label}</button>`).join('');
    $('steps').querySelectorAll('button').forEach(b=>b.onclick=()=>go(+b.dataset.step));
    $('eyebrow').textContent=`${att?'One head · exact arithmetic':'Decoder transformer · schematic'} / ${state.step+1} of ${labels.length}`;
    $('stage-title').textContent=(att?attentionTitles:transformerTitles)[state.step];
    $('legend').hidden=!att;
    $('tokens').hidden=!att;
    $('tokens').innerHTML=C.words.map((w,j)=>`<button data-query="${j}" aria-pressed="${state.query===j}" aria-label="Use ${w} at position ${j+1} as the query">${w}<small>${state.query===j?'↑ position being updated':`position ${j+1}`}</small></button>`).join('');
    $('tokens').querySelectorAll('button').forEach(b=>b.onclick=()=>setQuery(+b.dataset.query));
    const renderers=[renderInput,renderProjections,r=>renderMatch(r,false),r=>renderMatch(r,true),renderSoftmax,renderGather,renderAll];
    const desc=att?`Selected query: ${C.words[state.query]}. ${attentionTitles[state.step]}. ${state.step>=4?`Attention weights ${r.A[state.query].map(n=>n.toFixed(3)).join(', ')}. Head output ${vec(r.Z[state.query],3)}.`:''}`:transformerTitles[state.step];
    $('scene').innerHTML=`<title id="scene-title">${(att?attentionTitles:transformerTitles)[state.step]}</title><desc id="scene-description">${desc}</desc>${defs()}<g class="scene-entry">${att?renderers[state.step](r):renderTransformer(r)}</g>`;
    $('scene').setAttribute('role',att&&state.step===6?'group':'img');
    $('scene').querySelectorAll('[data-row]').forEach(g=>{
      const activate=()=>setQuery(+g.dataset.row);
      g.onclick=activate;g.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();activate();}};
    });
    $('visual-note').textContent=att ? [
      'Small, invented vectors make every operation visible.',
      'x has 3 coordinates; q, k and v have 2. The projections are shared across words.',
      'The lines show comparisons. The values have not moved yet.',
      'The diagonal is allowed: a position can always read itself.',
      'Rounded numbers are displayed; full precision is used in every calculation.',
      'Values travel from the source positions to the position being updated.',
      'Rows compute in parallel from the same input. This is one head, before Wₒ and the residual addition.'
    ][state.step] : 'Pre-norm architecture · information flows downward · moving dots trace the active computation';
    if(att && (state.equal || state.zeroValue)) $('visual-note').textContent = [state.equal ? 'Equal weights: all allowed logits are 0.' : '', state.zeroValue ? 'Value intervention: v(cat) = [0, 0]; Q and K are unchanged.' : ''].filter(Boolean).join(' ');
    $('explanation').innerHTML=att?explainAttention(r):explainTransformer();
    $('experiments').hidden=!att||state.step<3;
    $('zero-value').parentElement.hidden=state.step<5;
    $('equal').parentElement.hidden=state.step<4;
    $('inspect').hidden=!att;
    $('original').hidden=!att;
    $('provenance').textContent=att?'Hand-set teaching example · all attention calculations computed live':'Architecture schematic · no trained model or invented prediction probabilities';
    $('previous').disabled=state.step===0;
    $('next').disabled=state.step===stepCount()-1;
    $('play').textContent=state.playing?'Ⅱ Pause':'▶ Play';
    if($('zoom-out'))$('zoom-out').onclick=()=>{state.transformerStep=2;setView('transformer');};
    if($('zoom-in'))$('zoom-in').onclick=()=>setView('attention');
    if(restoreFocus)document.querySelector(restoreFocus)?.focus({preventScroll:true});
  }
  function table(rows,headers) {
    return `<table>${headers?`<thead><tr>${headers.map(h=>`<th>${h}</th>`).join('')}</tr></thead>`:''}<tbody>${rows.map(row=>`<tr>${row.map(v=>`<td>${typeof v==='number'?f(v,4):v}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
  }
  function inspectNumbers() {
    stop();const r=C.compute(state),i=state.query;
    const matrices=[['E: word embeddings',C.E],['P: position embeddings',C.P],['X = E + P',r.X],['W<sub>Q</sub>',C.WQ],['W<sub>K</sub>',C.WK],['W<sub>V</sub>',C.WV],['Q = X W<sub>Q</sub>',r.Q],['K = X W<sub>K</sub>',r.K],['V = X W<sub>V</sub>'+(state.zeroValue?' · cat value zeroed':''),r.V]];
    $('number-content').innerHTML=`<p class="details-copy">All weights below are hand-set, and the same projection matrix is applied to every input row. In a trained model these weights are learned. The position count (3) and input width (3) happen to match here; the query/key width is 2.</p><div class="matrices">${matrices.map(([name,rows])=>`<div class="matrix-card"><h3>${name}</h3>${table(rows)}</div>`).join('')}</div><h3>One projection, expanded: q(${C.words[i]}) = x(${C.words[i]}) W<sub>Q</sub></h3><p class="formula">${[0,1].map(k=>`${r.X[i].map((n,j)=>`${f(n)} × ${f(C.WQ[j][k])}`).join(' + ')} = ${f(r.Q[i][k])}`).join('<br>')}</p><h3>Selected row: ${C.words[i]}</h3><div class="table-scroll">${table(C.words.map((w,j)=>[w, r.scores[i][j],r.scaled[i][j],r.logits[i][j],r.exponents[i][j],r.A[i][j],vec(r.V[j],3),vec(r.V[j].map(v=>v*r.A[i][j]),4)]),['Source','Dot product','Scaled','Masked / override','Exponential','Weight','Value','Contribution'])}</div><p class="formula">z(${C.words[i]}) ≈ ${vec(r.Z[i],4)}</p><p class="details-copy">The exponentials subtract the maximum allowed logit for numerical stability. Future entries are exactly zero after softmax. Displayed values are rounded.</p>`;
    $('numbers').showModal();
  }
  $('attention-view').onclick=()=>setView('attention');
  $('transformer-view').onclick=()=>setView('transformer');
  $('previous').onclick=()=>go(state.step-1);
  $('next').onclick=()=>go(state.step+1);
  $('play').onclick=()=>{
    if(state.playing){stop();return;}
    if(state.step===stepCount()-1)state.step=0;
    state.playing=true;render();
    timer=setInterval(()=>{
      if(state.step<stepCount()-1)state.step++;
      if(state.step===stepCount()-1)stop();
      render();
    },4500);
  };
  $('reset').onclick=()=>{
    stop();Object.assign(state,{step:0,attentionStep:0,transformerStep:0,query:2,mask:true,scale:true,equal:false,zeroValue:false});
    ['mask','scale','equal','zero-value'].forEach(id=>$(id).checked=id==='mask'||id==='scale');render();
  };
  [['mask','mask'],['scale','scale'],['equal','equal'],['zero-value','zeroValue']].forEach(([id,key])=>$(id).onchange=()=>{stop();state[key]=$(id).checked;render();});
  $('inspect').onclick=inspectNumbers;
  $('close-numbers').onclick=()=>$('numbers').close();
  $('numbers').addEventListener('click',e=>{if(e.target===$('numbers')){const b=$('numbers').getBoundingClientRect();if(e.clientX<b.left||e.clientX>b.right||e.clientY<b.top||e.clientY>b.bottom)$('numbers').close();}});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  new ResizeObserver(()=>{
    const bounds=$('scene').getBoundingClientRect();
    $('tokens').style.width=Math.min(bounds.width,bounds.height*1040/360)+'px';
  }).observe($('scene'));
  render();
})();
