/* Escape exits an embedded control without moving the current slide. */
window.addEventListener('message', event => {
  if(event.data?.type !== 'cpsy-exit-interactive') return;
  const frame=[...document.querySelectorAll('.interactive-figure iframe')].find(f=>f.contentWindow===event.source);
  if(!frame)return;
  document.body.tabIndex=-1;
  document.body.focus({preventScroll:true});
});

/* Adjacent slides retain the same measured plot and its user-controlled state. */
(() => {
 const states=new Map();
 const frames=()=>[...document.querySelectorAll('iframe[data-continuity]')];
 window.addEventListener('message',event=>{
  const frame=frames().find(f=>f.contentWindow===event.source);if(!frame)return;
  const key=frame.dataset.continuity;
  if(event.data?.type==='cpsy-coordinate-ready'){
   if(states.has(key))frame.contentWindow.postMessage({type:'cpsy-coordinate-restore',state:states.get(key)},'*');
  }else if(event.data?.type==='cpsy-coordinate-state'){
   const state=event.data.state;
   if(!state||!Number.isInteger(state.selected)||state.selected<0||state.selected>5||!Number.isFinite(state.gain)||state.gain<.25||state.gain>1.5||!Number.isInteger(state.geometryStep)||state.geometryStep<0||state.geometryStep>2)return;
   states.set(key,state);
   frames().filter(f=>f!==frame&&f.dataset.continuity===key).forEach(f=>f.contentWindow.postMessage({type:'cpsy-coordinate-restore',state},'*'));
  }
 });
 frames().forEach(frame=>{frame.addEventListener('load',()=>frame.contentWindow.postMessage({type:'cpsy-coordinate-request'},'*'));frame.contentWindow.postMessage({type:'cpsy-coordinate-request'},'*')});
})();

/* Stop lecture film audio when leaving its slide or hiding the page. */
(() => {
 const pauseHiddenFilms=()=>document.querySelectorAll('iframe[src*="hubel_wiesel.html"]').forEach(frame=>{
  const slide=frame.closest('section');
  if(document.hidden || (location.hash && slide?.id !== location.hash.slice(1)))
   frame.contentWindow?.postMessage({type:'cpsy-media-pause'},'*');
 });
 window.addEventListener('hashchange',pauseHiddenFilms);
 document.addEventListener('visibilitychange',pauseHiddenFilms);
})();

/* One source location persists through RGB, grayscale and vector-profile views. */
(() => {
 let state=null;
 const frames=()=>[...document.querySelectorAll('iframe[data-image-values]')];
 window.addEventListener('message',event=>{
  const frame=frames().find(f=>f.contentWindow===event.source);if(!frame)return;
  if(event.data?.type==='cpsy-image-values-ready'){
   if(state)frame.contentWindow.postMessage({type:'cpsy-image-values-restore',state},'*');
  }else if(event.data?.type==='cpsy-image-values-state'){
   const s=event.data.state;if(!s||![s.sourceRow,s.sourceCol].every(x=>Number.isInteger(x)&&x>=0&&x<224))return;
   state={sourceRow:s.sourceRow,sourceCol:s.sourceCol};
   frames().filter(f=>f!==frame).forEach(f=>f.contentWindow.postMessage({type:'cpsy-image-values-restore',state},'*'));
  }
 });
 frames().forEach(frame=>{frame.addEventListener('load',()=>frame.contentWindow.postMessage({type:'cpsy-image-values-request'},'*'));frame.contentWindow.postMessage({type:'cpsy-image-values-request'},'*')});
})();
