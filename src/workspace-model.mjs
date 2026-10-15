// Shared navigation and data normalization, independent of the React application.
export const GROUPS = [
  { id: 'today', label: 'Today', description: 'Your research agenda', items: [['heatmap', 'Portfolio heat map'], ['dashboard', 'Portfolio health'], ['alerts', 'Alerts']] },
  { id: 'companies', label: 'Companies', description: 'Coverage & investment views', items: [['overview', 'Company overview'], ['stockanalysis', 'Stock analysis'], ['portfolio', 'Investment thesis'], ['thesisimports', 'Import & review thesis'], ['review', 'Investment review']] },
  { id: 'library', label: 'Library', description: 'Evidence & conversations', items: [['summary', 'Summaries'], ['summarylab', 'Summary Lab'], ['research', 'Research documents'], ['explain', 'Explain a document'], ['chat', 'Ask Charlie']] },
  { id: 'create', label: 'Create', description: 'Turn research into an output', items: [['deepdive', 'Research brief'], ['onepager', 'Investment one-pager'], ['catalysts', 'Catalyst notes'], ['meetingprep', 'Meeting preparation'], ['formats', 'Thesis templates'], ['slides', 'Presentations'], ['studio', 'Creative studio']] },
  { id: 'activity', label: 'Activity', description: 'Work in progress & recent results', items: [] },
  { id: 'automations', label: 'Automations', description: 'Monitor, investigate & update', items: [['desk', 'Research desk'], ['pipeline', 'Research pipeline'], ['pipelinelab', 'Research Pipeline Lab'], ['analysts', 'Analyst team'], ['agents', 'Research agents'], ['feed', 'Podcast feed']] },
];
export const VIEWS = [...GROUPS.flatMap(g => [g.id, ...g.items.map(i => i[0])]), 'settings'];
export const viewLabel = id => GROUPS.find(g => g.id === id)?.label || GROUPS.flatMap(g => g.items).find(i => i[0] === id)?.[1] || 'Settings';
export const viewGroup = id => GROUPS.find(g => g.id === id || g.items.some(i => i[0] === id));
export const SECTIONS = {
  desk: ['work','ready','overview','command','collection','history','priorities','earnings','evidence','inbox','queue','launch','runs','playbooks'],
  stockanalysis: ['snapshot','stock-analysis','research','committee','case','model','changes','evidence','reviews','signals','evolution'],
};
export const validSection = (view, section) => SECTIONS[view]?.includes(section) ? section : '';
// A work link selects a saved item; it never authorizes starting or resuming it.
export function validWork(view,section,value='') {
 if(view==='meetingprep'&&!section&&/^meeting:[1-9][0-9]{0,9}$/.test(value)&&Number(String(value).split(':')[1])<=2147483647)return value;
 const allowed={'stockanalysis:stock-analysis':['assignment','analysis'],'stockanalysis:research':['research'],'stockanalysis:committee':['committee'],'stockanalysis:reviews':['decision'],'desk:collection':['collection'],'desk:evidence':['amendment'],'pipelinelab:':['labnote']};
 const [kind,id,...rest]=String(value).split(':');
 return !rest.length&&allowed[`${view}:${section}`]?.includes(kind)&&/^(?:[a-f0-9]{32}|[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12})$/i.test(id||'')?`${kind}:${id.toLowerCase()}`:'';
}
export const workId=(value,kind)=>value?.startsWith(kind+':')?value.slice(kind.length+1):'';
export function readRoute(hash = '') {
  const params = new URLSearchParams(hash.replace(/^#/, ''));
  const view = params.get('view');
  const section = validSection(view, params.get('section'));
  const work = validWork(view,section,params.get('work'));
  return { ...(work ? {work} : {}), ...(section ? {section} : {}), view: VIEWS.includes(view) ? view : 'today', ticker: (params.get('ticker') || '').trim().toUpperCase().slice(0,32) };
}
export function routeHash(view, ticker = '', section = '', work = '') {
  const p = new URLSearchParams({ view: VIEWS.includes(view) ? view : 'today' });
  if (ticker) p.set('ticker', ticker);
  if (validSection(view,section)) p.set('section',section);
  if (validWork(view,section,work)) p.set('work',validWork(view,section,work));
  return '#' + p.toString();
}
export function parseTimestamp(value) {
  if (value == null || value === '') return null;
  if (value instanceof Date) return Number.isNaN(value.getTime()) ? null : value;
  let s = value;
  if (typeof value === 'string') {
    s = value.trim();
    // PostgreSQL naive timestamps are UTC; RFC dates already have a zone.
    if (/^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}/.test(s) && !/(Z|[+-]\d{2}(:?\d{2})?)$/i.test(s)) s = s.replace(' ', 'T') + 'Z';
    if (/^\d{4}-\d{2}-\d{2}$/.test(s)) s += 'T12:00:00Z';
  }
  const d = new Date(s);
  return Number.isNaN(d.getTime()) ? null : d;
}
export function companyIndex(analyses = [], overviews = []) {
  const rows = new Map();
  for (const item of [...overviews, ...analyses]) {
    const ticker = String(item.ticker || '').trim().toUpperCase();
    if (!ticker) continue;
    const old = rows.get(ticker) || { ticker, company: ticker, hasThesis: false, hasOverview: false };
    rows.set(ticker, { ...old, company: item.company || item.companyName || item.name || old.company,
      hasThesis: old.hasThesis || analyses.includes(item), hasOverview: old.hasOverview || overviews.includes(item),
      updatedAt: item.updatedAt || item.createdAt || old.updatedAt });
  }
  return [...rows.values()].sort((a,b) => a.ticker.localeCompare(b.ticker));
}

export function selectedProjectSlide(project, currentProjectId, currentSlideNumber) {
  const slides = project.slides || [];
  return (project.id === currentProjectId && slides.find(s => s.slide_number === currentSlideNumber)) || slides[0] || null;
}

// Calendar dates are compared in the investor's timezone, not the device timezone.
export function upcomingMeetings(meetings,now=new Date()) {
 const today=now.toLocaleDateString('en-CA',{timeZone:'America/New_York'});
 return (Array.isArray(meetings)?meetings:[]).filter(m=>{
  const day=m?.meeting_date;
  return typeof day==='string'&&/^\d{4}-\d{2}-\d{2}$/.test(day)&&Number.isFinite(Date.parse(day))&&new Date(day).toISOString().slice(0,10)===day&&day>=today&&validWork('meetingprep','',`meeting:${m.id}`)&&typeof m.ticker==='string'&&m.ticker.trim();
 }).sort((a,b)=>a.meeting_date.localeCompare(b.meeting_date)||String(a.ticker).localeCompare(String(b.ticker))||Number(a.id)-Number(b.id));
}
