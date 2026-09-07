function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
import * as React from 'react';
import { GROUPS, viewLabel, viewGroup, companyIndex, parseTimestamp } from './workspace-model.mjs';
var {
  useState,
  useEffect,
  useRef
} = React;
function Icon({
  name,
  ...props
}) {
  var paths = {
    today: 'M3 4h18v16H3z M3 9h18 M8 2v4 M16 2v4 M7 13h3 M14 13h3 M7 17h3',
    companies: 'M4 21V5h10v16 M14 11h6v10 M8 9h2 M8 13h2 M8 17h2 M2 21h20',
    library: 'M3 4h5v16H3z M10 4h5v16h-5z M17 5l4-1 3 15-4 1z',
    create: 'M12 3v18 M3 12h18',
    automations: 'M13 2L4 14h7l-1 8 10-13h-7z',
    search: 'M10 3a7 7 0 1 0 0 14 7 7 0 0 0 0-14 M15 15l6 6',
    menu: 'M4 6h16 M4 12h16 M4 18h16',
    close: 'M6 6l12 12 M18 6L6 18',
    arrow: 'M5 12h14 M14 7l5 5-5 5',
    settings: 'M4 7h16 M4 17h16 M9 4v6 M15 14v6',
    chat: 'M3 4h18v13H8l-5 4z'
  };
  return /*#__PURE__*/React.createElement("svg", _extends({
    width: "20",
    height: "20",
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: "1.5",
    strokeLinecap: "round",
    strokeLinejoin: "round",
    "aria-hidden": "true"
  }, props), /*#__PURE__*/React.createElement("path", {
    d: paths[name] || paths.today
  }));
}
export function WorkspaceShell({
  active,
  onNavigate,
  themeControl,
  local,
  health,
  ticker,
  onCompany
}) {
  var [menu, setMenu] = useState(false);
  var [search, setSearch] = useState('');
  var dialog = useRef(null);
  var trigger = useRef(null);
  var group = viewGroup(active);
  var go = id => {
    setMenu(false);
    setSearch('');
    onNavigate(id);
  };
  useEffect(() => {
    if (!menu) return;
    var before = document.activeElement;
    dialog.current?.querySelector('input')?.focus();
    var key = e => {
      if (e.key === 'Escape') {
        setMenu(false);
        return;
      }
      if (e.key !== 'Tab') return;
      var elements = [...dialog.current.querySelectorAll('button,input,a')].filter(x => !x.disabled && x.getClientRects().length);
      var first = elements[0],
        last = elements[elements.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last?.focus();
      }
      if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first?.focus();
      }
    };
    document.addEventListener('keydown', key);
    return () => {
      document.removeEventListener('keydown', key);
      before?.focus();
    };
  }, [menu]);
  useEffect(() => {
    var handler = e => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setMenu(x => !x);
      }
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, []);
  var unavailable = health && (health.status === 'stale' || health.agentSeenEver === false);
  return /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("a", {
    className: "workspace-skip",
    href: "#workspace-content",
    onClick: e => {
      e.preventDefault();
      document.getElementById('workspace-content')?.focus();
    }
  }, "Skip to content"), /*#__PURE__*/React.createElement("aside", {
    className: "workspace-sidebar",
    "aria-label": "Workspace navigation"
  }, /*#__PURE__*/React.createElement("button", {
    className: "workspace-brand",
    onClick: () => go('today'),
    "aria-label": "Charlie home"
  }, /*#__PURE__*/React.createElement("span", {
    className: "workspace-monogram"
  }, "C"), /*#__PURE__*/React.createElement("span", null, "Charlie", /*#__PURE__*/React.createElement("small", null, "RESEARCH WORKSPACE"))), /*#__PURE__*/React.createElement("button", {
    ref: trigger,
    className: "workspace-search",
    onClick: () => setMenu(true)
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "search"
  }), /*#__PURE__*/React.createElement("span", null, "Find a workspace"), /*#__PURE__*/React.createElement("kbd", null, "\u2318 K")), /*#__PURE__*/React.createElement("nav", {
    "aria-label": "Main navigation"
  }, GROUPS.map(g => /*#__PURE__*/React.createElement("div", {
    key: g.id,
    className: "workspace-nav-group"
  }, /*#__PURE__*/React.createElement("button", {
    className: 'workspace-nav-item ' + (group?.id === g.id ? 'is-active' : ''),
    onClick: () => go(g.id),
    "aria-current": active === g.id ? 'page' : undefined
  }, /*#__PURE__*/React.createElement(Icon, {
    name: g.id
  }), g.label, /*#__PURE__*/React.createElement("span", {
    className: "nav-indicator"
  })), group?.id === g.id && /*#__PURE__*/React.createElement("div", {
    className: "workspace-subnav"
  }, g.items.map(([id, label]) => /*#__PURE__*/React.createElement("button", {
    key: id,
    onClick: () => go(id),
    "aria-current": active === id ? 'page' : undefined
  }, label)))))), /*#__PURE__*/React.createElement("div", {
    className: "workspace-sidebar-foot"
  }, /*#__PURE__*/React.createElement("button", {
    onClick: () => go('chat')
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "chat"
  }), "Ask Charlie"), /*#__PURE__*/React.createElement("button", {
    onClick: () => go('settings'),
    "aria-current": active === 'settings' ? 'page' : undefined
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "settings"
  }), "Settings"), /*#__PURE__*/React.createElement("div", {
    className: "workspace-theme"
  }, themeControl), /*#__PURE__*/React.createElement("small", null, "Evidence. Conviction. Perspective."))), /*#__PURE__*/React.createElement("header", {
    className: "workspace-header"
  }, /*#__PURE__*/React.createElement("div", {
    className: "workspace-breadcrumb"
  }, /*#__PURE__*/React.createElement("span", null, group?.label || 'Workspace'), /*#__PURE__*/React.createElement("span", null, "/"), /*#__PURE__*/React.createElement("strong", null, active === group?.id ? group.description : viewLabel(active))), /*#__PURE__*/React.createElement("div", {
    className: "workspace-header-actions"
  }, ticker && /*#__PURE__*/React.createElement("button", {
    className: "workspace-ticker",
    onClick: () => onCompany(ticker)
  }, ticker), /*#__PURE__*/React.createElement("span", {
    className: "workspace-local"
  }, local ? 'TEST DATA' : 'LIVE DATA'), /*#__PURE__*/React.createElement("details", {
    className: "workspace-health"
  }, /*#__PURE__*/React.createElement("summary", null, /*#__PURE__*/React.createElement("span", {
    className: 'workspace-status-dot ' + (unavailable ? 'is-warning' : '')
  }), /*#__PURE__*/React.createElement("span", null, local ? 'Local session' : health ? unavailable ? 'Agent offline' : 'Agent connected' : 'Checking agent')), /*#__PURE__*/React.createElement("div", {
    role: "status"
  }, /*#__PURE__*/React.createElement("strong", null, local ? 'Development environment' : unavailable ? 'Local agent unavailable' : 'Research services'), /*#__PURE__*/React.createElement("p", null, local ? 'You are using the local backend. The Mac agent reports to production, so its local heartbeat may be stale.' : unavailable ? 'Tasks that need files on your Mac may wait until the agent reconnects. You can continue reading saved research.' : 'Your Mac agent is connected.'), /*#__PURE__*/React.createElement("button", {
    onClick: () => go('settings')
  }, "View settings ", /*#__PURE__*/React.createElement(Icon, {
    name: "arrow"
  })))), /*#__PURE__*/React.createElement("button", {
    className: "workspace-menu-trigger",
    onClick: () => setMenu(true),
    "aria-label": "Open workspace menu"
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "menu"
  })))), /*#__PURE__*/React.createElement("nav", {
    className: "workspace-mobile-nav",
    "aria-label": "Mobile navigation"
  }, GROUPS.slice(0, 4).map(g => /*#__PURE__*/React.createElement("button", {
    key: g.id,
    onClick: () => go(g.id),
    "aria-current": group?.id === g.id ? 'page' : undefined
  }, /*#__PURE__*/React.createElement(Icon, {
    name: g.id
  }), /*#__PURE__*/React.createElement("span", null, g.label))), /*#__PURE__*/React.createElement("button", {
    onClick: () => setMenu(true),
    "aria-label": "All workspaces"
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "menu"
  }), /*#__PURE__*/React.createElement("span", null, "More"))), menu && /*#__PURE__*/React.createElement("div", {
    className: "workspace-dialog-scrim",
    onClick: () => setMenu(false)
  }, /*#__PURE__*/React.createElement("section", {
    className: "workspace-command",
    role: "dialog",
    "aria-modal": "true",
    "aria-labelledby": "workspace-menu-title",
    ref: dialog,
    onClick: e => e.stopPropagation()
  }, /*#__PURE__*/React.createElement("div", {
    className: "workspace-command-head"
  }, /*#__PURE__*/React.createElement("h2", {
    id: "workspace-menu-title"
  }, "Your workspaces"), /*#__PURE__*/React.createElement("button", {
    onClick: () => setMenu(false),
    "aria-label": "Close workspace menu"
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "close"
  }))), /*#__PURE__*/React.createElement("label", {
    className: "workspace-command-search"
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "search"
  }), /*#__PURE__*/React.createElement("input", {
    placeholder: "Find a tool or workflow\u2026",
    "aria-label": "Search workspaces",
    value: search,
    onChange: e => setSearch(e.target.value)
  })), /*#__PURE__*/React.createElement("div", {
    className: "workspace-command-results"
  }, GROUPS.map(g => {
    var items = [[g.id, g.label], ...g.items].filter(i => i[1].toLowerCase().includes(search.toLowerCase()));
    return items.length ? /*#__PURE__*/React.createElement("div", {
      key: g.id
    }, /*#__PURE__*/React.createElement("h3", null, g.label), items.map(([id, label]) => /*#__PURE__*/React.createElement("button", {
      key: id,
      onClick: () => go(id)
    }, /*#__PURE__*/React.createElement("span", null, label), active === id ? /*#__PURE__*/React.createElement("small", null, "Current") : /*#__PURE__*/React.createElement(Icon, {
      name: "arrow"
    })))) : null;
  }), 'settings'.includes(search.toLowerCase()) && /*#__PURE__*/React.createElement("button", {
    onClick: () => go('settings')
  }, "Settings"), !GROUPS.some(g => [[g.id, g.label], ...g.items].some(i => i[1].toLowerCase().includes(search.toLowerCase()))) && !'settings'.includes(search.toLowerCase()) && /*#__PURE__*/React.createElement("p", null, "No matching workspace. Try \u201Creview\u201D or \u201Csummary\u201D.")))));
}
var dateLabel = v => parseTimestamp(v)?.toLocaleDateString('en-US', {
  month: 'short',
  day: 'numeric',
  timeZone: 'America/New_York'
}) || 'Date unavailable';
function PageHeading({
  eyebrow,
  title,
  children,
  action
}) {
  return /*#__PURE__*/React.createElement("div", {
    className: "workspace-page-heading"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, eyebrow), /*#__PURE__*/React.createElement("h1", null, title), /*#__PURE__*/React.createElement("p", {
    className: "workspace-lead"
  }, children)), action);
}
function Empty({
  children
}) {
  return /*#__PURE__*/React.createElement("p", {
    className: "workspace-empty"
  }, children);
}
export function TodayWorkspace({
  analyses,
  overviews,
  summaries,
  alerts,
  meetings,
  onNavigate,
  onSummary,
  onCompany
}) {
  var companies = companyIndex(analyses, overviews);
  var recent = [...summaries].sort((a, b) => (parseTimestamp(b.createdAt)?.getTime() || 0) - (parseTimestamp(a.createdAt)?.getTime() || 0)).slice(0, 5);
  var pending = alerts.filter(a => !['dismissed', 'done', 'archived'].includes(a.status));
  var upcoming = (meetings || []).filter(m => m.meeting_date && m.meeting_date >= new Date().toLocaleDateString('en-CA')).sort((a, b) => a.meeting_date.localeCompare(b.meeting_date)).slice(0, 3);
  return /*#__PURE__*/React.createElement("div", {
    className: "workspace-page"
  }, /*#__PURE__*/React.createElement(PageHeading, {
    eyebrow: new Date().toLocaleDateString('en-US', {
      weekday: 'long',
      month: 'long',
      day: 'numeric'
    }),
    title: "A clearer view of what matters.",
    action: /*#__PURE__*/React.createElement("button", {
      className: "workspace-primary",
      onClick: () => onNavigate('create')
    }, "Create research ", /*#__PURE__*/React.createElement(Icon, {
      name: "arrow"
    }))
  }, "Revisit your companies, follow the evidence, and move your research forward."), /*#__PURE__*/React.createElement("div", {
    className: "workspace-metrics"
  }, [[companies.length, 'Companies in your library', 'companies'], [summaries.length, 'Saved source summaries', 'library'], [pending.length, 'Alerts to review', 'alerts']].map(([n, label, id]) => /*#__PURE__*/React.createElement("button", {
    key: id,
    onClick: () => onNavigate(id)
  }, /*#__PURE__*/React.createElement("strong", null, n), /*#__PURE__*/React.createElement("span", null, label), /*#__PURE__*/React.createElement(Icon, {
    name: "arrow"
  })))), /*#__PURE__*/React.createElement("div", {
    className: "workspace-home-grid"
  }, /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "workspace-section-heading"
  }, /*#__PURE__*/React.createElement("h2", null, "Continue your research"), /*#__PURE__*/React.createElement("button", {
    onClick: () => onNavigate('library')
  }, "Open library \u2197")), recent.length ? recent.map(s => /*#__PURE__*/React.createElement("button", {
    className: "workspace-document-row",
    key: s.id,
    onClick: () => onSummary(s)
  }, /*#__PURE__*/React.createElement("span", {
    className: "workspace-document-icon"
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "library"
  })), /*#__PURE__*/React.createElement("span", null, /*#__PURE__*/React.createElement("strong", null, s.title || 'Untitled document'), /*#__PURE__*/React.createElement("small", null, s.topic || 'General', " \xB7 ", s.docType || s.sourceType || 'Research')), /*#__PURE__*/React.createElement("time", null, dateLabel(s.createdAt)), /*#__PURE__*/React.createElement(Icon, {
    name: "arrow"
  }))) : /*#__PURE__*/React.createElement(Empty, null, "Your saved research will appear here. Start by adding a document to the Library.")), /*#__PURE__*/React.createElement("div", {
    className: "workspace-home-side"
  }, /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "NEXT STEP"), /*#__PURE__*/React.createElement("h2", null, "What are you working on?"), [['companies', 'Revisit an investment view'], ['meetingprep', 'Prepare for a management meeting'], ['explain', 'Understand a difficult document']].map(([id, label]) => /*#__PURE__*/React.createElement("button", {
    className: "workspace-link-row",
    key: id,
    onClick: () => onNavigate(id)
  }, label, /*#__PURE__*/React.createElement(Icon, {
    name: "arrow"
  })))), /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("h2", null, "Upcoming meetings"), upcoming.length ? upcoming.map(m => /*#__PURE__*/React.createElement("button", {
    className: "workspace-link-row",
    key: m.id,
    onClick: () => onNavigate('meetingprep')
  }, m.ticker, " ", /*#__PURE__*/React.createElement("time", null, dateLabel(m.meeting_date)))) : /*#__PURE__*/React.createElement(Empty, null, "No upcoming meetings in this workspace. Your saved meeting preparation is available in Create.")))), /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("div", {
    className: "workspace-section-heading"
  }, /*#__PURE__*/React.createElement("h2", null, "Company quick access"), /*#__PURE__*/React.createElement("button", {
    onClick: () => onNavigate('companies')
  }, "All companies \u2197")), /*#__PURE__*/React.createElement("div", {
    className: "workspace-company-chips"
  }, companies.slice(0, 12).map(c => /*#__PURE__*/React.createElement("button", {
    key: c.ticker,
    onClick: () => onCompany(c.ticker)
  }, c.ticker, /*#__PURE__*/React.createElement("small", null, c.company !== c.ticker ? c.company : c.hasThesis ? 'Thesis saved' : 'Overview saved')))), !companies.length && /*#__PURE__*/React.createElement(Empty, null, "Build your coverage by researching your first company.")));
}
export function CompaniesWorkspace({
  analyses,
  overviews,
  ticker,
  onCompany,
  onNavigate
}) {
  var [query, setQuery] = useState('');
  var companies = companyIndex(analyses, overviews),
    selected = companies.find(c => c.ticker === ticker);
  var visible = companies.filter(c => (c.ticker + ' ' + c.company).toLowerCase().includes(query.toLowerCase()));
  return /*#__PURE__*/React.createElement("div", {
    className: "workspace-page"
  }, /*#__PURE__*/React.createElement(PageHeading, {
    eyebrow: "YOUR COVERAGE",
    title: "Companies",
    action: /*#__PURE__*/React.createElement("button", {
      className: "workspace-primary",
      onClick: () => onNavigate('create')
    }, "Research a company ", /*#__PURE__*/React.createElement(Icon, {
      name: "arrow"
    }))
  }, "One place to return to the evidence and investment view behind each name."), ticker && /*#__PURE__*/React.createElement("section", {
    className: "workspace-company-focus"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "CURRENT COMPANY"), /*#__PURE__*/React.createElement("h2", null, ticker, " ", /*#__PURE__*/React.createElement("span", null, selected?.company !== ticker ? selected?.company : '')), /*#__PURE__*/React.createElement("p", null, selected ? [selected.hasOverview && 'Overview available', selected.hasThesis && 'Thesis available'].filter(Boolean).join(' · ') : 'Start research or add evidence for this company.')), /*#__PURE__*/React.createElement("div", {
    className: "workspace-company-actions"
  }, [['overview', 'Overview'], ['portfolio', 'Thesis'], ['summary', 'Evidence'], ['review', 'Review'], ['onepager', 'One-pager']].map(([id, label]) => /*#__PURE__*/React.createElement("button", {
    key: id,
    onClick: () => onCompany(ticker, id)
  }, label, /*#__PURE__*/React.createElement(Icon, {
    name: "arrow"
  }))))), /*#__PURE__*/React.createElement("div", {
    className: "workspace-list-toolbar"
  }, /*#__PURE__*/React.createElement("label", {
    className: "workspace-filter"
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "search"
  }), /*#__PURE__*/React.createElement("input", {
    "aria-label": "Search companies",
    placeholder: "Search ticker or company\u2026",
    value: query,
    onChange: e => setQuery(e.target.value)
  })), /*#__PURE__*/React.createElement("span", null, visible.length, " companies")), /*#__PURE__*/React.createElement("div", {
    className: "workspace-company-grid"
  }, visible.map(c => /*#__PURE__*/React.createElement("button", {
    className: 'workspace-company-card ' + (ticker === c.ticker ? 'is-selected' : ''),
    key: c.ticker,
    onClick: () => onCompany(c.ticker)
  }, /*#__PURE__*/React.createElement("span", {
    className: "workspace-company-card-top"
  }, /*#__PURE__*/React.createElement("strong", null, c.ticker), /*#__PURE__*/React.createElement(Icon, {
    name: "arrow"
  })), /*#__PURE__*/React.createElement("span", null, c.company), /*#__PURE__*/React.createElement("small", null, c.hasThesis ? 'Thesis + research' : 'Company overview')))), !visible.length && /*#__PURE__*/React.createElement(Empty, null, "No matching companies. Try another name, or create new research."));
}
export function LibraryWorkspace({
  summaries,
  onSummary,
  onNavigate
}) {
  var [query, setQuery] = useState('');
  var [filter, setFilter] = useState('all');
  var matches = [...summaries].filter(s => `${s.title} ${s.topic} ${s.docType} ${s.sourceType}`.toLowerCase().includes(query.toLowerCase()) && (filter === 'all' || s.sourceType === filter)).sort((a, b) => (parseTimestamp(b.createdAt)?.getTime() || 0) - (parseTimestamp(a.createdAt)?.getTime() || 0));
  return /*#__PURE__*/React.createElement("div", {
    className: "workspace-page"
  }, /*#__PURE__*/React.createElement(PageHeading, {
    eyebrow: "THE EVIDENCE BEHIND THE VIEW",
    title: "Research library",
    action: /*#__PURE__*/React.createElement("button", {
      className: "workspace-primary",
      onClick: () => onNavigate('summary')
    }, "Add or manage documents ", /*#__PURE__*/React.createElement(Icon, {
      name: "arrow"
    }))
  }, "Search your saved summaries by company, title, or source type."), /*#__PURE__*/React.createElement("div", {
    className: "workspace-list-toolbar"
  }, /*#__PURE__*/React.createElement("label", {
    className: "workspace-filter"
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "search"
  }), /*#__PURE__*/React.createElement("input", {
    "aria-label": "Search library",
    placeholder: "Search documents, companies, topics\u2026",
    value: query,
    onChange: e => setQuery(e.target.value)
  })), /*#__PURE__*/React.createElement("select", {
    "aria-label": "Filter by source",
    value: filter,
    onChange: e => setFilter(e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: "all"
  }, "All sources"), [...new Set(summaries.map(s => s.sourceType).filter(Boolean))].map(v => /*#__PURE__*/React.createElement("option", {
    key: v,
    value: v
  }, v))), /*#__PURE__*/React.createElement("span", null, matches.length, " documents")), /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel"
  }, matches.slice(0, 100).map(s => /*#__PURE__*/React.createElement("button", {
    className: "workspace-document-row",
    key: s.id,
    onClick: () => onSummary(s)
  }, /*#__PURE__*/React.createElement("span", {
    className: "workspace-document-icon"
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "library"
  })), /*#__PURE__*/React.createElement("span", null, /*#__PURE__*/React.createElement("strong", null, s.title || 'Untitled document'), /*#__PURE__*/React.createElement("small", null, s.topic || 'General', " \xB7 ", s.docType || s.sourceType || 'Document')), /*#__PURE__*/React.createElement("time", null, dateLabel(s.createdAt)), /*#__PURE__*/React.createElement(Icon, {
    name: "arrow"
  }))), !matches.length && /*#__PURE__*/React.createElement(Empty, null, "No documents match this search. Try a company ticker or clear the source filter."), matches.length > 100 && /*#__PURE__*/React.createElement("p", {
    className: "workspace-empty"
  }, "Showing the 100 most recent matches. Narrow your search to find older documents.")), /*#__PURE__*/React.createElement("button", {
    className: "workspace-link-row",
    onClick: () => onNavigate('research')
  }, "Browse research documents and folders ", /*#__PURE__*/React.createElement(Icon, {
    name: "arrow"
  })));
}
var CREATIONS = [['deepdive', 'Company research brief', 'A fresh research pass with a one-page brief, two-page report, and investment memo.', 'Ticker · Web research'], ['review', 'Investment review', 'Revisit the thesis, scenarios, and what changed using your stored evidence.', 'Ticker · Saved sources'], ['meetingprep', 'Meeting preparation', 'Turn documents into focused questions for management.', 'Company · Source documents'], ['onepager', 'Investment one-pager', 'A visual reference for the stock debate and key signposts.', 'Ticker · Existing + web research'], ['slides', 'Presentation', 'Build and refine a slide narrative from your research.', 'Documents · Editable slides'], ['formats', 'Thesis export', 'Present a saved thesis as a scorecard, IC memo, or another decision format.', 'Saved thesis · Template'], ['explain', 'Document explanation', 'Understand dense language, assumptions, and financial terminology.', 'Text · PDF · Screenshots'], ['studio', 'Creative studio', 'Explore infographics, mind maps, and learning materials.', 'Sources · Output format']];
export function CreateWorkspace({
  ticker,
  onTicker,
  onNavigate
}) {
  return /*#__PURE__*/React.createElement("div", {
    className: "workspace-page"
  }, /*#__PURE__*/React.createElement(PageHeading, {
    eyebrow: "FROM EVIDENCE TO OUTPUT",
    title: "What would you like to create?"
  }, "Choose the outcome first. You can refine sources, depth, and model in the workspace."), /*#__PURE__*/React.createElement("label", {
    className: "workspace-create-context"
  }, "Company context ", /*#__PURE__*/React.createElement("input", {
    "aria-label": "Company context ticker",
    placeholder: "Ticker (optional)",
    value: ticker,
    onChange: e => onTicker(e.target.value.toUpperCase())
  }), /*#__PURE__*/React.createElement("span", null, "Carried into company research tools")), /*#__PURE__*/React.createElement("div", {
    className: "workspace-create-grid"
  }, CREATIONS.map(([id, title, description, meta], i) => /*#__PURE__*/React.createElement("button", {
    key: id,
    onClick: () => onNavigate(id),
    className: "workspace-create-card"
  }, /*#__PURE__*/React.createElement("span", {
    className: "workspace-create-number"
  }, "0", i + 1), /*#__PURE__*/React.createElement("h2", null, title), /*#__PURE__*/React.createElement("p", null, description), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("small", null, meta), /*#__PURE__*/React.createElement(Icon, {
    name: "arrow"
  }))))));
}
export function AutomationsWorkspace({
  onNavigate,
  local
}) {
  return /*#__PURE__*/React.createElement("div", {
    className: "workspace-page"
  }, /*#__PURE__*/React.createElement(PageHeading, {
    eyebrow: "KEEP YOUR RESEARCH CURRENT",
    title: "Automations"
  }, "Follow signals, review proposed work, and manage research that runs in the background."), local && /*#__PURE__*/React.createElement("p", {
    className: "workspace-notice"
  }, "Local development session. Mac-agent jobs are connected to production; review configuration before starting a run."), /*#__PURE__*/React.createElement("div", {
    className: "workspace-create-grid"
  }, [['desk', 'Research desk', 'Coordinate company teams, review aging theses, and track research runs.'], ['analysts', 'Analyst team', 'Review pending investigations, coverage, and earnings activities.'], ['pipeline', 'Research pipeline', 'Generate notes and refresh investment theses across your universe.'], ['feed', 'Podcast feed', 'Filter new episodes and material mentions of companies you cover.'], ['agents', 'Research agents', 'Configure multi-perspective analysis and catalyst investigations.']].map(([id, title, description]) => /*#__PURE__*/React.createElement("button", {
    className: "workspace-create-card",
    key: id,
    onClick: () => onNavigate(id)
  }, /*#__PURE__*/React.createElement(Icon, {
    name: "automations"
  }), /*#__PURE__*/React.createElement("h2", null, title), /*#__PURE__*/React.createElement("p", null, description), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("small", null, "Open workspace"), /*#__PURE__*/React.createElement(Icon, {
    name: "arrow"
  }))))));
}
// Server documents contain global CSS. A script-free frame prevents them from
// changing the host application while preserving their authored tables/styles.
export function ResearchDocument({
  html,
  title = 'Investment review'
}) {
  var frame = useRef(null),
    observer = useRef(null);
  var [height, setHeight] = useState(800);
  useEffect(() => () => observer.current?.disconnect(), []);
  var loaded = () => {
    observer.current?.disconnect();
    var doc = frame.current?.contentDocument;
    if (!doc) return;
    var measure = () => setHeight(Math.ceil(doc.documentElement.getBoundingClientRect().height) + 24);
    measure();
    observer.current = new ResizeObserver(measure);
    observer.current.observe(doc.documentElement);
  };
  var source = `<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><style>html{background:#fff;color:#1e293b;}body{margin:0;padding:24px;box-sizing:border-box;color:#1e293b;}html,body{height:auto;min-height:0;}img{max-width:100%;}table{max-width:100%;}*{box-sizing:border-box;}@media(max-width:600px){body{padding:16px;}table{font-size:11px!important;}}</style></head><body>${html || '<p>No document is available for this review.</p>'}</body></html>`;
  return /*#__PURE__*/React.createElement("iframe", {
    className: "workspace-document-frame",
    ref: frame,
    title: title,
    sandbox: "allow-same-origin",
    srcDoc: source,
    onLoad: loaded,
    style: {
      height
    }
  });
}