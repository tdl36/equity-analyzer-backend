// Shared navigation and data normalization, independent of the React application.
export const GROUPS = [
  { id: 'today', label: 'Today', description: 'Your research agenda', items: [['dashboard', 'Portfolio health'], ['alerts', 'Alerts']] },
  { id: 'companies', label: 'Companies', description: 'Coverage & investment views', items: [['overview', 'Company overview'], ['portfolio', 'Investment thesis'], ['review', 'Investment review']] },
  { id: 'library', label: 'Library', description: 'Evidence & conversations', items: [['summary', 'Summaries'], ['research', 'Research documents'], ['explain', 'Explain a document'], ['chat', 'Ask Charlie']] },
  { id: 'create', label: 'Create', description: 'Turn research into an output', items: [['deepdive', 'Research brief'], ['onepager', 'Investment one-pager'], ['meetingprep', 'Meeting preparation'], ['formats', 'Thesis templates'], ['slides', 'Presentations'], ['studio', 'Creative studio']] },
  { id: 'automations', label: 'Automations', description: 'Monitor, investigate & update', items: [['desk', 'Research desk'], ['pipeline', 'Research pipeline'], ['analysts', 'Analyst team'], ['agents', 'Research agents'], ['feed', 'Podcast feed']] },
];
export const VIEWS = [...GROUPS.flatMap(g => [g.id, ...g.items.map(i => i[0])]), 'settings'];
export const viewLabel = id => GROUPS.find(g => g.id === id)?.label || GROUPS.flatMap(g => g.items).find(i => i[0] === id)?.[1] || 'Settings';
export const viewGroup = id => GROUPS.find(g => g.id === id || g.items.some(i => i[0] === id));
export function readRoute(hash = '') {
  const params = new URLSearchParams(hash.replace(/^#/, ''));
  const view = params.get('view');
  return { view: VIEWS.includes(view) ? view : 'today', ticker: (params.get('ticker') || '').trim().toUpperCase().slice(0,32) };
}
export function routeHash(view, ticker = '') {
  const p = new URLSearchParams({ view: VIEWS.includes(view) ? view : 'today' });
  if (ticker) p.set('ticker', ticker);
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
