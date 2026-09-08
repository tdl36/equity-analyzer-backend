import { ResearchChat } from './research-chat';
import { ResearchAutomationControl } from './research-automation';
import { ThesisAmendments } from './thesis-amendments';
import { SavedResearchContext } from './saved-research-context';
import * as React from 'react';
var {
  useState,
  useEffect,
  useRef
} = React;
var date = value => value ? String(value).slice(0, 10) : 'Undated';
export function EvidenceWorkspace({
  api,
  analyses,
  onCompany
}) {
  var [ticker, setTicker] = useState('DE'),
    [input, setInput] = useState('DE');
  var [data, setData] = useState(null),
    [error, setError] = useState(''),
    [loading, setLoading] = useState(true);
  var [selected, setSelected] = useState(null),
    [refresh, setRefresh] = useState(0);
  var inspector = useRef(null);
  var inspect = claim => {
    setSelected(claim);
    requestAnimationFrame(() => inspector.current?.scrollIntoView({
      block: 'nearest',
      behavior: 'smooth'
    }));
  };
  useEffect(() => {
    var controller = new AbortController();
    var current = true;
    setLoading(true);
    setError('');
    setData(null);
    setSelected(null);
    var timer = setTimeout(() => controller.abort(), 20000);
    fetch(`${api}/api/research/evidence/${encodeURIComponent(ticker)}`, {
      signal: controller.signal
    }).then(async r => {
      if (!r.ok) throw new Error(r.status === 401 ? 'Sign in to view saved research.' : `Could not load research (${r.status}).`);
      return r.json();
    }).then(value => {
      if (current) setData(value);
    }).catch(e => {
      if (current) setError(e.name === 'AbortError' ? 'The request timed out. Please retry.' : e.message);
    }).finally(() => {
      clearTimeout(timer);
      if (current) setLoading(false);
    });
    return () => {
      current = false;
      clearTimeout(timer);
      controller.abort();
    };
  }, [api, ticker, refresh]);
  var review = data?.current,
    prior = data?.prior,
    claims = review?.evidence?.claims || [];
  var sources = review?.evidence?.sources || [];
  var pick = path => {
    var claim = claims.find(c => c.path === path);
    inspect(claim || {
      statement: 'No source passage was recorded for this item.',
      evidence: []
    });
  };
  var showCompany = () => onCompany(ticker, 'portfolio');
  return /*#__PURE__*/React.createElement("section", {
    className: "evidence-workspace"
  }, /*#__PURE__*/React.createElement("div", {
    className: "evidence-intro"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "RESEARCH / EVIDENCE & CHANGES"), /*#__PURE__*/React.createElement("h2", null, "Read the change. Inspect the evidence."), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Read your saved thesis, inspect available documents, and compare investment reviews when a baseline exists.")), /*#__PURE__*/React.createElement("form", {
    className: "evidence-search",
    onSubmit: e => {
      e.preventDefault();
      var tk = input.trim().toUpperCase();
      if (/^[A-Z0-9][A-Z0-9.\-]{0,19}$/.test(tk)) {
        setTicker(tk);
        setRefresh(x => x + 1);
      } else setError('Enter a valid ticker.');
    }
  }, /*#__PURE__*/React.createElement("label", {
    htmlFor: "evidence-ticker"
  }, "Company ticker"), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("input", {
    id: "evidence-ticker",
    list: "evidence-companies",
    value: input,
    onChange: e => setInput(e.target.value),
    maxLength: 20
  }), /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    type: "submit"
  }, "Open \u2192")), /*#__PURE__*/React.createElement("datalist", {
    id: "evidence-companies"
  }, [...new Set((analyses || []).map(a => a.ticker).filter(Boolean))].map(t => /*#__PURE__*/React.createElement("option", {
    key: t,
    value: t
  }))))), /*#__PURE__*/React.createElement(ResearchAutomationControl, {
    api: api
  }), !loading && !error && data && /*#__PURE__*/React.createElement(ThesisAmendments, {
    key: ticker,
    api: api,
    ticker: ticker,
    context: data,
    onApplied: () => setRefresh(x => x + 1)
  }), !loading && !error && (data?.savedThesis || review) && /*#__PURE__*/React.createElement("details", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("summary", null, "Discuss this research with an analyst"), /*#__PURE__*/React.createElement(ResearchChat, {
    key: `${ticker}:${refresh}`,
    api: api,
    onApplied: () => setRefresh(x => x + 1),
    context: {
      ticker,
      type: review ? 'review' : 'thesis',
      content: JSON.stringify(review?.state || data.savedThesis)
    }
  })), loading ? /*#__PURE__*/React.createElement("p", {
    role: "status",
    className: "workspace-empty"
  }, "Loading saved research for ", ticker, "\u2026") : error ? /*#__PURE__*/React.createElement("div", {
    className: "workspace-error",
    role: "alert"
  }, error, " ", /*#__PURE__*/React.createElement("button", {
    onClick: () => setRefresh(x => x + 1)
  }, "Retry")) : !review ? /*#__PURE__*/React.createElement(SavedResearchContext, {
    key: ticker,
    data: data,
    ticker: ticker,
    onCompany: onCompany
  }) : /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("details", {
    className: "workspace-panel evidence-context-toggle"
  }, /*#__PURE__*/React.createElement("summary", null, "Saved company thesis & available documents"), /*#__PURE__*/React.createElement(SavedResearchContext, {
    key: ticker,
    data: data,
    ticker: ticker,
    onCompany: onCompany
  })), /*#__PURE__*/React.createElement("div", {
    className: "evidence-summary"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("strong", null, ticker), /*#__PURE__*/React.createElement("span", null, "Review \xB7 ", date(review.createdAt)), /*#__PURE__*/React.createElement("span", null, prior ? `Compared with ${date(prior.createdAt)}` : 'First saved review · no baseline')), /*#__PURE__*/React.createElement("span", {
    className: "evidence-badge"
  }, review.quality.status === 'checks_passed' ? 'Automated checks passed · analyst review required' : 'Needs review'), /*#__PURE__*/React.createElement("button", {
    onClick: showCompany
  }, "Open company \u2197")), /*#__PURE__*/React.createElement("div", {
    className: "evidence-layout"
  }, /*#__PURE__*/React.createElement("main", {
    className: "evidence-main"
  }, /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "01 / INVESTMENT JUDGMENT"), /*#__PURE__*/React.createElement("h3", null, "The current thesis"), (review.state.thesis || []).map((text, i) => /*#__PURE__*/React.createElement("button", {
    className: "evidence-claim",
    key: i,
    onClick: () => pick(`thesis.${i}`)
  }, /*#__PURE__*/React.createElement("span", null, String(i + 1).padStart(2, '0')), /*#__PURE__*/React.createElement("div", null, text, /*#__PURE__*/React.createElement("small", null, claims.find(c => c.path === `thesis.${i}`)?.status === 'passage_matched' ? 'Inspect matching passage →' : 'Source evidence needed →')))), !review.state.thesis?.length && /*#__PURE__*/React.createElement("p", null, "No thesis statements recorded.")), /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "02 / WHAT CHANGED"), /*#__PURE__*/React.createElement("h3", null, "Saved thesis comparison"), prior ? /*#__PURE__*/React.createElement("div", {
    className: "evidence-compare"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h4", null, "Previous \xB7 ", date(prior.createdAt)), (prior.state.thesis || []).map((t, i) => /*#__PURE__*/React.createElement("p", {
    key: i
  }, t))), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("h4", null, "Current \xB7 ", date(review.createdAt)), (review.state.thesis || []).map((t, i) => /*#__PURE__*/React.createElement("p", {
    key: i
  }, t)))) : /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "No previous investment review exists. Changes cannot be independently compared yet."), prior && (review.state.changes || []).map((c, i) => /*#__PURE__*/React.createElement("article", {
    className: "evidence-change",
    key: i
  }, /*#__PURE__*/React.createElement("h4", null, c.item), /*#__PURE__*/React.createElement("div", {
    className: "evidence-compare"
  }, /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, "Model-described prior"), c.prior || 'Not specified'), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("small", null, "Model-described current"), c.current || 'Not specified')), /*#__PURE__*/React.createElement("p", null, c.implication), /*#__PURE__*/React.createElement("button", {
    onClick: () => pick(`changes.${i}`)
  }, "Inspect supporting evidence \u2192"))), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "These are saved review versions and model-described changes, not a verified event feed or amendments to your thesis.")), /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "03 / FACT REGISTER"), /*#__PURE__*/React.createElement("h3", null, "Claims and provenance"), claims.filter(c => !c.path.startsWith('thesis.') && !c.path.startsWith('changes.')).map(c => /*#__PURE__*/React.createElement("button", {
    key: c.path,
    className: "evidence-fact",
    onClick: () => inspect(c)
  }, /*#__PURE__*/React.createElement("span", null, c.type?.replaceAll('_', ' ') || 'Unclassified'), /*#__PURE__*/React.createElement("strong", null, c.statement), /*#__PURE__*/React.createElement("small", null, c.status === 'passage_matched' ? 'Passage matched' : 'Needs evidence', " \u2192"))), !claims.some(c => !c.path.startsWith('thesis.') && !c.path.startsWith('changes.')) && /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "This review has no captured fact evidence. Older reports are not retroactively treated as verified."))), /*#__PURE__*/React.createElement("aside", {
    className: "evidence-sidebar",
    "aria-label": "Quality and source evidence"
  }, /*#__PURE__*/React.createElement("section", {
    className: "workspace-panel"
  }, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "RESEARCH READINESS"), /*#__PURE__*/React.createElement("h3", null, review.quality.matchedCount, " / ", review.quality.claimCount, " passages matched"), /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, review.quality.meaning), review.quality.issues.length > 0 ? /*#__PURE__*/React.createElement("ul", null, review.quality.issues.map((issue, i) => /*#__PURE__*/React.createElement("li", {
    key: i
  }, issue))) : /*#__PURE__*/React.createElement("p", null, "No automated blockers recorded. Review assumptions and conclusions before use."), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Document coverage"), /*#__PURE__*/React.createElement("p", null, review.documentsRead.length, " read \xB7 ", review.documentsNotRead.length, " not read"), review.documentsRead.map((name, i) => /*#__PURE__*/React.createElement("p", {
    key: i
  }, name)), review.documentsNotRead.map((name, i) => /*#__PURE__*/React.createElement("p", {
    key: i
  }, "Not read: ", name)))), /*#__PURE__*/React.createElement("section", {
    ref: inspector,
    className: "workspace-panel evidence-source",
    "aria-live": "polite"
  }, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "SOURCE INSPECTOR"), /*#__PURE__*/React.createElement("h3", null, selected ? 'Supporting passages' : 'Inspect a claim'), selected ? /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", null, selected.statement), selected.evidence.length ? selected.evidence.map((e, i) => {
    var source = sources.find(s => s.id === e.sourceId);
    return /*#__PURE__*/React.createElement("article", {
      key: i
    }, /*#__PURE__*/React.createElement("strong", null, source?.filename || 'Unknown source'), /*#__PURE__*/React.createElement("small", null, e.status === 'passage_matched' ? 'Exact passage matched in captured extraction' : 'Passage could not be matched'), /*#__PURE__*/React.createElement("blockquote", null, e.excerpt || 'No excerpt captured.'), /*#__PURE__*/React.createElement("p", {
      className: "desk-explainer"
    }, e.status === 'passage_matched' ? 'A text match is not a conclusion check. Assess the surrounding context in the original document.' : 'Do not rely on this quotation until checked against the original.'));
  }) : /*#__PURE__*/React.createElement("p", null, "No source excerpt was captured. Open the company research to consult the originals.")) : /*#__PURE__*/React.createElement("p", {
    className: "desk-explainer"
  }, "Select a thesis statement, change, or fact to inspect its source passage and matching status."))))));
}