'use strict';
// Explicit fictional mode; no trained-model evidence is displayed here.
function initializeDemo() {
  const titles={property:'Property simulator',churn:'Churn simulator',disease:'Classification simulator'};
  const notice=state.summary.demo_notice;
  $('title').textContent=titles[state.project];
  document.title=titles[state.project]+' | Synthetic demo';
  $('project-tag').textContent='SYNTHETIC DEMO';
  $('subtitle').textContent='Try single records, CSV batches and downloads using entirely fictional data.';
  document.querySelector('.model span').textContent='DEMO ENGINE';
  $('winner').textContent='Invented formula';
  $('notice').textContent=notice;
  $('notice').classList.add('warning');
  $('metrics').hidden=true;
  $('scenario').hidden=true;
  $('interpretation').textContent=notice+' Blank inputs use the example values. No validation metrics or uncertainty intervals apply.';
  document.querySelector('footer').textContent=notice;
  document.querySelector('[data-panel="evaluate"]').textContent='Demo explanation';
  const explanation=el('article',undefined,'card');
  explanation.append(el('h2','How this fictional demo works'),el('p',notice));
  const formulas={property:'Demo value = 10,000 + 1,200 × floor area + 5,000 × rooms. Units and prices are fictional.',churn:'Demo score = ((1 − months / 120) + (1 − service score / 10)) / 2. This is not an estimated churn probability.',disease:'Demo score = (signal A + signal B) / 20. Both signals are invented and have no clinical meaning.'};
  explanation.append(el('p',formulas[state.project]),el('p','No dataset or trained artifact is loaded. No real-world performance is claimed. Restart without --demo only after reproducing the trained artifacts.'));
  $('evaluate').replaceChildren(explanation);
  for(const spec of state.schema) {
    const label=el('label',spec.name.replaceAll('_',' ')), input=el('input');
    input.type='number'; input.step='any'; input.min=spec.min; input.max=spec.max;
    input.id='field-'+spec.name; input.name=spec.name; label.htmlFor=input.id;
    label.append(input,el('small',spec.unit+' · range '+spec.min+' to '+spec.max+' · blank = example value'));
    $('fields').append(label);
  }
  fill();
  if(state.project!=='property') {
    $('threshold-box').hidden=false;
    $('threshold').value=.5; $('threshold-value').textContent='0.50';
    document.querySelector('#threshold-box p').textContent='This cutoff changes fictional flags only; it has no measured accuracy or clinical meaning.';
  }
}
function renderDemoResult(data) {
  const box=$('result'), result=data.results[0];
  box.replaceChildren(el('h2','Synthetic demo output'));
  box.append(el('div',state.project==='property'?number(result.prediction):percent(result.probability),'prediction'));
  box.append(el('p',state.project==='property'?'Fictional value in arbitrary demo units. No prediction interval applies.':'Fictional score, not a calibrated probability.'));
  if(state.project!=='property') box.append(el('p',(result.prediction?'Demo flag on':'Demo flag off')+' at cutoff '+result.threshold.toFixed(2)));
  warnings(box,data.warnings);
  box.scrollIntoView({behavior:'smooth',block:'center'});
}
