/* Controlled key–value retrieval examples. No language model or human data. */
(function(root){
 'use strict';
 const locations=['desk','shelf','drawer'];
 const target={id:'target',color:'red',shape:'triangle',location:'desk',key:[1,0,1,0]};
 const redCircle={id:'red-circle',color:'red',shape:'circle',location:'shelf',key:[1,0,0,1]};
 const blueTriangle={id:'blue-triangle',color:'blue',shape:'triangle',location:'drawer',key:[0,1,1,0]};
 const blueCircle={id:'blue-circle',color:'blue',shape:'circle',location:'drawer',key:[0,1,0,1]};
 function compute({scenario='match',sharpness=6,distractors=24,shapeCue=false,restore=false,reverse=false}={}){
  let records=scenario==='clutter'?[target,...Array.from({length:distractors},(_,i)=>({...redCircle,id:'distractor-'+i}))]:scenario==='missing'?[...(restore?[target]:[]),redCircle,blueCircle]:[target,redCircle,blueTriangle];
  if(reverse)records=records.slice().reverse();
  const q=scenario==='ambiguous'&&!shapeCue?[1,0,0,0]:[1,0,1,0];
  const raw=records.map(r=>q.reduce((s,v,j)=>s+v*r.key[j],0));
  const logits=raw.map(s=>sharpness*s/Math.sqrt(4));
  const max=Math.max(...logits),exp=logits.map(s=>Math.exp(s-max));
  const total=exp.reduce((a,b)=>a+b,0),weights=exp.map(e=>e/total);
  const values=records.map(r=>locations.map(l=>+(r.location===l)));
  const output=locations.map((_,j)=>weights.reduce((s,w,i)=>s+w*values[i][j],0));
  const peak=Math.max(...output),winners=locations.filter((_,j)=>Math.abs(output[j]-peak)<1e-10);
  return{records,q,raw,logits,weights,values,output,winners};
 }
 const api={compute,locations};
 if(typeof module!=='undefined'&&module.exports)module.exports=api;
 else root.RetrievalAttention=api;
 if(typeof document==='undefined')return;
 const $=id=>document.getElementById(id);
 const state={scenario:'match',sharpness:6,distractors:24,shapeCue:false,restore:false,reverse:false,revealed:false,guess:null};
 const cases=[['match','1 · A useful cue'],['ambiguous','2 · An incomplete cue'],['clutter','3 · Many distractors'],['missing','4 · No correct source']];
 const titles={match:'Where is the red triangle?',ambiguous:'What if the query forgets the shape?',clutter:'Can many weak matches beat one good match?',missing:'What if the answer is not available?'};
 const setups={match:'Three stored objects, three locations. The query encodes both color and shape.',ambiguous:'The question still asks for the red triangle. This query encodes only “red.”',clutter:'One source has the right object. Many other sources match its color, but not its shape.',missing:'The red triangle was on the desk. Its source is now missing from the memory bank.'};
 const color={red:'#B5396B',blue:'#3E6D8E',ink:'#1E293B',value:'#3F7A6B',line:'#d8d7ce',paper:'#FAF9F5'};
 const pct=x=>(100*x).toFixed(1)+'%';
 const v=x=>'['+x.map(n=>Number(n.toFixed(3))).join(', ')+']';
 const text=(x,y,t,cls='',anchor='middle')=>`<text x="${x}" y="${y}" text-anchor="${anchor}" class="${cls}">${t}</text>`;
 const box=(x,y,w,h,stroke=color.line,fill='none',radius=10)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${radius}" fill="${fill}" stroke="${stroke}"/>`;
 const shape=(r,x,y,size=22)=>r.shape==='triangle'?`<path d="M${x},${y-size} L${x-size},${y+size*.8} L${x+size},${y+size*.8} Z" fill="${color[r.color]}"/>`:`<circle cx="${x}" cy="${y}" r="${size}" fill="${color[r.color]}"/>`;
 function draw(r){
  const grouped=state.scenario==='clutter';
  const groups=grouped?[{record:target,indices:r.records.map((x,i)=>x.id==='target'?i:-1).filter(i=>i>=0)},{record:redCircle,indices:r.records.map((x,i)=>x.id!=='target'?i:-1).filter(i=>i>=0)}]:r.records.map((record,i)=>({record,indices:[i]}));
  if(grouped&&state.reverse)groups.reverse();
  const centers=groups.map((_,j)=>(j+.5)*1080/groups.length);
  let s='<title id="scene-title">Cue-based retrieval</title><desc id="scene-desc">The question asks where the red triangle is. '+(state.revealed?locations.map((l,j)=>l+' receives '+pct(r.output[j])).join('; '):'Predict which location will dominate before revealing the calculation.')+'</desc>';
  s+=box(337,8,406,61,color.red,'#b5396b08');
  s+=text(540,32,'QUERY · what to match','small');
  s+=text(540,57,r.q[2]?'red + triangle':'red · shape missing','cue bold');
  groups.forEach(({record,indices},j)=>{
   const x=centers[j],mass=indices.reduce((s,i)=>s+r.weights[i],0),n=indices.length;
   const width=grouped?380:Math.min(300,1080/groups.length-28);
   s+=`<path d="M540,70 C540,95 ${x},85 ${x},113" fill="none" stroke="${color.line}" stroke-width="2"/>`;
   s+=box(x-width/2,119,width,153,color.line,color.paper);
   if(n===1){s+=shape(record,x-70,159,20)+text(x-30,154,record.color+' '+record.shape,'bold','start')+text(x-30,179,'key · description','tiny','start');}
   else{
    for(let k=0;k<Math.min(n,40);k++)s+=shape(record,x-145+(k%10)*32,140+Math.floor(k/10)*15,5);
    s+=text(x,210,n+' similar sources','bold');
   }
   s+=text(x,249,'value → '+record.location,'value bold');
   if(state.revealed){
    s+=box(x-width/2,282,width,25,color.line,color.paper,5);
    s+=`<rect x="${x-width/2}" y="282" width="${width*mass}" height="25" rx="5" fill="${color.value}" class="bar"/>`;
    s+=text(x,335,pct(mass)+(n>1?' in total':''),'value large');
    if(n>1)s+=text(x,360,pct(r.weights[indices[0]])+' per source','small');
    const d=`M${x},${n>1?368:346} C${x},382 540,374 540,397`;
    s+=`<path d="${d}" fill="none" stroke="${color.value}" stroke-width="${2+mass*5}"/>`;
    if(!matchMedia('(prefers-reduced-motion:reduce)').matches)s+=`<circle r="5" fill="${color.value}" opacity="0"><animateMotion path="${d}" dur="1.5s" fill="freeze"/><animate attributeName="opacity" values="0;1;1;0" dur="1.5s" fill="freeze"/></circle>`;
   }else s+=text(x,319,'share = ?','small');
  });
  if(state.revealed){
   s+=box(145,400,790,38,color.value,'#3f7a6b08');
   s+=text(540,427,'RETURNED VECTOR · '+locations.map((l,j)=>l+' '+pct(r.output[j])).join(' / '),'value small');
  }else s+=text(540,413,'Each source supplies a location. Attention mixes their contributions.','small');
  $('scene').innerHTML=s;
 }
 function explanation(r){
  const winner=r.winners.length===3?'All three locations tie':r.winners.length===2?'Desk and shelf tie':r.winners[0][0].toUpperCase()+r.winners[0].slice(1)+' dominates';
  const success=r.winners.length===1&&r.winners[0]==='desk';
  let detail='';
  if(state.scenario==='match')detail=success?'The query distinguishes the target. Its value contributes most to the result.':'The weighting is too broad to isolate the target. Sharper matches can help here.';
  if(state.scenario==='ambiguous')detail=state.shapeCue?'Restoring shape makes the query distinguish the two red objects.':'Turning up sharpness cannot break an exact tie. The query needs a distinguishing feature.';
  if(state.scenario==='clutter')detail=success?'The best source still wins, but the distractors collectively take some of its share.':'Each distractor is a weaker match. Together they supply more content than the correct source.';
  if(state.scenario==='missing')detail=state.restore?'Restoring the source makes the correct location available again.':'Every available value names the wrong location. Sharper attention can concentrate on a wrong answer.';
  if(state.sharpness===0)detail='Every source has equal weight. The cue is ignored, so repeated locations contribute more copies of the same value.';
  const chosen=state.guess?`You predicted ${state.guess==='tie'?'a tie':state.guess}.`:'No prediction selected.';
  return `<p class="verdict">${winner}</p><p>${detail}</p><p class="prediction-note">${chosen} Shares measure the returned vector, not correctness.</p>`;
 }
 function render(){
  const r=compute(state);
  $('case-label').textContent=cases.find(([id])=>id===state.scenario)[1];
  $('question').textContent=titles[state.scenario];
  $('setup').textContent=setups[state.scenario];
  if(state.scenario==='ambiguous'&&state.shapeCue)$('setup').textContent='The question and the query now both specify red + triangle.';
  if(state.scenario==='missing'&&state.restore)$('setup').textContent='The matching source has been restored: red triangle → desk.';
  document.querySelectorAll('[data-case]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.case===state.scenario));
  document.querySelectorAll('[data-guess]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.guess===state.guess));
  $('challenge').hidden=state.revealed;$('reveal').hidden=state.revealed;$('result').hidden=!state.revealed;
  $('result').innerHTML=state.revealed?explanation(r):'';
  $('clutter-control').hidden=state.scenario!=='clutter';$('shape-control').hidden=state.scenario!=='ambiguous';$('restore-control').hidden=state.scenario!=='missing';
  $('sharpness').value=state.sharpness;$('sharpness-value').textContent=state.sharpness===0?'0 · equal weights':state.sharpness;
  $('distractors').value=state.distractors;$('distractor-value').textContent=state.distractors;
  $('shape-cue').checked=state.shapeCue;$('restore').checked=state.restore;
  const takeaways={match:'A useful match routes relevant content. At sharpness 0, every source has equal weight.',ambiguous:'A sharper selection cannot recover a distinction the query does not encode.',clutter:'Attention weights compete for a fixed total. Many weaker matches can collectively dominate.',missing:'A normalized attention row must distribute its weight somewhere. This head has no “none of the above” source.'};
  $('takeaway').textContent=state.revealed?takeaways[state.scenario]:'Make a prediction, then reveal. The bars are shares of retrieved content—not probabilities of being correct.';
  draw(r);
 }
 function reset(scenario=state.scenario){Object.assign(state,{scenario,sharpness:6,distractors:24,shapeCue:false,restore:false,reverse:false,revealed:false,guess:null});render();}
 $('cases').innerHTML=cases.map(([id,label])=>`<button data-case="${id}" aria-pressed="false">${label}</button>`).join('');
 document.querySelectorAll('[data-case]').forEach(b=>b.onclick=()=>reset(b.dataset.case));
 document.querySelectorAll('[data-guess]').forEach(b=>b.onclick=()=>{state.guess=b.dataset.guess;render();});
 $('reveal').onclick=()=>{state.revealed=true;render();};
 $('reset').onclick=()=>reset();
 $('sharpness').oninput=e=>{state.sharpness=+e.target.value;render();};
 $('distractors').oninput=e=>{state.distractors=+e.target.value;render();};
 $('shape-cue').onchange=e=>{state.shapeCue=e.target.checked;render();};
 $('restore').onchange=e=>{state.restore=e.target.checked;render();};
 $('shuffle').onclick=()=>{state.reverse=!state.reverse;render();$('takeaway').textContent='Reordering the same available key–value pairs leaves this fixed query’s result unchanged. No position features are used here.';};
 $('numbers').onclick=()=>{
  const r=compute(state);$('math-content').innerHTML=`<p>Keys encode [red, blue, triangle, circle]. Values encode [desk, shelf, drawer]. The query is <b>${v(r.q)}</b>. These vectors are supplied by hand; nothing here parses language or learns.</p><div class="math-formula">scoreⱼ = β (q · kⱼ) / √4<br>a = softmax(scores)<br>z = Σⱼ aⱼ vⱼ</div><p>β = ${state.sharpness} is the adjustable match sharpness. β = 0 gives uniform weights. The square-root divisor uses the key width, 4. All percentages come from full-precision computation.</p><div class="table-wrap"><table><thead><tr><th>Source</th><th>Key</th><th>Value</th><th>Score</th><th>Weight</th></tr></thead><tbody>${r.records.map((record,i)=>`<tr><td>${record.color} ${record.shape}</td><td>${v(record.key)}</td><td>${v(r.values[i])}</td><td>${r.logits[i].toFixed(2)}</td><td>${pct(r.weights[i])}</td></tr>`).join('')}</tbody></table></div><p>Output ≈ ${v(r.output)}. The output happens to sum to one because the supplied values are one-hot location codes. General value vectors need not sum to one.</p><p>A fixed query is invariant to permutations of its available key–value pairs. This is not a claim that a causal transformer has no order information. Likewise, the missing-source example is not a model of all hallucinations: a whole trained model can learn to abstain.</p>`;$('math').showModal();
 };
 $('close-math').onclick=()=>$('math').close();
 const params=new URLSearchParams(location.search);if(params.get('embed')==='1')document.body.classList.add('embed');
 if(cases.some(([id])=>id===params.get('case')))state.scenario=params.get('case');
 render();
})(typeof globalThis!=='undefined'?globalThis:this);
