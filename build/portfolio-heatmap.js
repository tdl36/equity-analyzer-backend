import * as React from 'react';
import { groupedLayout, formatReturn, tileColor, parseHoldings } from './portfolio-heatmap-model.mjs';
var periods = [['1d', '1 day'], ['1w', '1 week'], ['1m', '1 month'], ['3m', '3 months'], ['6m', '6 months'], ['ytd', 'YTD'], ['1y', '1 year']];
var blank = () => ({
  name: 'My portfolio',
  asOf: new Date().toLocaleDateString('en-CA'),
  holdings: []
});
var number = v => Number(v).toLocaleString(undefined, {
  maximumFractionDigits: 2
});
async function apiCall(url, options = {}) {
  var r = await fetch(url, {
    ...options,
    signal: options.signal ? AbortSignal.any([options.signal, AbortSignal.timeout(60000)]) : AbortSignal.timeout(60000)
  });
  var d = await r.json();
  if (!r.ok) throw Error(d.error || 'Request failed. Please retry.');
  return d;
}
function download(name, body, type = 'text/csv') {
  var url = URL.createObjectURL(new Blob([body], {
    type
  }));
  var a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
var csvCell = v => '"' + String(v ?? '').replaceAll('"', '""') + '"';
export function PortfolioHeatmap({
  api,
  onOpen
}) {
  var [saved, setSaved] = React.useState(null),
    [draft, setDraft] = React.useState(blank),
    [revision, setRevision] = React.useState(0),
    [editing, setEditing] = React.useState(false);
  var [error, setError] = React.useState(''),
    [notice, setNotice] = React.useState(''),
    [saving, setSaving] = React.useState(false),
    [period, setPeriod] = React.useState('1d');
  var [market, setMarket] = React.useState(null),
    [loading, setLoading] = React.useState(false),
    [refresh, setRefresh] = React.useState(0),
    [filter, setFilter] = React.useState(''),
    [sector, setSector] = React.useState('All sectors');
  var [group, setGroup] = React.useState(true),
    [accessible, setAccessible] = React.useState(false),
    [view, setView] = React.useState(() => window.innerWidth < 700 ? 'list' : 'map'),
    [selected, setSelected] = React.useState(null),
    [paste, setPaste] = React.useState('');
  var [universe, setUniverse] = React.useState('portfolio'),
    [index, setIndex] = React.useState(null),
    [indexLoading, setIndexLoading] = React.useState(false),
    [indexError, setIndexError] = React.useState(''),
    [indexRetry, setIndexRetry] = React.useState(0),
    [progress, setProgress] = React.useState(0);
  var active = universe === 'portfolio' ? saved : index?.universe === universe ? index : null;
  var [width, setWidth] = React.useState(1000);
  var mapRef = React.useRef(null);
  var load = async () => {
    setError('');
    try {
      var d = await apiCall(api + '/api/portfolio/heatmap/holdings');
      var body = d.body?.holdings ? d.body : blank();
      setSaved(body);
      setDraft(body);
      setRevision(d.revision);
      setEditing(!body.holdings.length);
    } catch (e) {
      setError(e.message);
    }
  };
  React.useEffect(() => {
    load();
  }, [api]);
  React.useEffect(() => {
    if (!mapRef.current) return;
    var o = new ResizeObserver(entries => setWidth(Math.max(280, entries[0].contentRect.width)));
    o.observe(mapRef.current);
    return () => o.disconnect();
  }, [view, active]);
  React.useEffect(() => {
    setIndex(null);
    setIndexError('');
    setSector('All sectors');
    setFilter('');
    setSelected(null);
    if (universe === 'portfolio') {
      setIndexLoading(false);
      return;
    }
    var controller = new AbortController();
    setIndexLoading(true);
    apiCall(api + '/api/portfolio/heatmap/universe/' + universe, {
      signal: controller.signal
    }).then(d => {
      if (!controller.signal.aborted) setIndex(d.body);
    }).catch(e => {
      if (!controller.signal.aborted) setIndexError(e.message);
    }).finally(() => {
      if (!controller.signal.aborted) setIndexLoading(false);
    });
    return () => controller.abort();
  }, [api, universe, indexRetry]);
  React.useEffect(() => {
    setMarket(null);
    setSelected(null);
    setProgress(0);
    setLoading(false);
    if (!active?.holdings.length) return;
    var controller = new AbortController();
    setLoading(true);
    setError('');
    (async () => {
      var failed = 0;
      var _loop = async function () {
          if (controller.signal.aborted) return {
            v: void 0
          };
          var batch = active.holdings.slice(start, start + 40);
          try {
            var d = await apiCall(api + '/api/portfolio/heatmap/returns', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json'
              },
              body: JSON.stringify({
                tickers: batch.map(r => r.ticker),
                period
              }),
              signal: controller.signal
            });
            if (controller.signal.aborted) return {
              v: void 0
            };
            setMarket(previous => ({
              ...d,
              quotes: {
                ...previous?.quotes,
                ...d.quotes
              }
            }));
          } catch (e) {
            if (controller.signal.aborted) return {
              v: void 0
            };
            failed += batch.length;
          }
          setProgress(Math.min(start + batch.length, active.holdings.length));
        },
        _ret;
      for (var start = 0; start < active.holdings.length; start += 40) {
        _ret = await _loop();
        if (_ret) return _ret.v;
      }
      if (!controller.signal.aborted) {
        setLoading(false);
        if (failed) setError(`Price requests failed for ${failed} holdings. Available returns are shown; use Refresh prices to retry.`);
      }
    })();
    return () => controller.abort();
  }, [api, active, period, refresh]);
  var save = async () => {
    setError('');
    setSaving(true);
    try {
      var d = await apiCall(api + '/api/portfolio/heatmap/holdings', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          ...draft,
          revision
        })
      });
      setSaved(d.body);
      setDraft(d.body);
      setRevision(d.revision);
      setEditing(false);
      setNotice('Holdings saved across your Charlie devices.');
    } catch (e) {
      setError(e.message);
    } finally {
      setSaving(false);
    }
  };
  var importText = text => {
    try {
      var holdings = parseHoldings(text);
      setDraft(d => ({
        ...d,
        holdings
      }));
      setNotice(`Imported ${holdings.length} rows into your draft. Check the holdings date, then save.`);
      setError('');
    } catch (e) {
      setError(e.message);
    }
  };
  var rows = (active?.holdings || []).map(r => ({
    ...r,
    ...(market?.quotes?.[r.ticker] || {})
  }));
  var visible = rows.filter(r => (sector === 'All sectors' || r.sector === sector) && `${r.ticker} ${r.company}`.toLowerCase().includes(filter.toLowerCase()));
  var gross = rows.reduce((s, r) => s + Math.abs(r.weight), 0),
    covered = rows.filter(r => Number.isFinite(r.changePct)).reduce((s, r) => s + Math.abs(r.weight), 0);
  var groups = groupedLayout(visible, group, width, Math.max(440, Math.min(660, width * .58)));
  var chosen = rows.find(r => r.ticker === selected);
  var edit = (i, key, value) => setDraft(d => ({
    ...d,
    holdings: d.holdings.map((r, j) => i === j ? {
      ...r,
      [key]: value
    } : r)
  }));
  return /*#__PURE__*/React.createElement("main", {
    className: "portfolio-map"
  }, /*#__PURE__*/React.createElement("style", null, styles), /*#__PURE__*/React.createElement("header", {
    className: "ph-heading"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("p", {
    className: "ph-eyebrow"
  }, "PORTFOLIO / MARKET PULSE"), /*#__PURE__*/React.createElement("h1", null, active?.name || (universe === 'portfolio' ? 'Portfolio heat map' : 'Market heat map')), /*#__PURE__*/React.createElement("p", null, universe === 'portfolio' ? 'Your positions, sized by exposure. See where the movement is concentrated.' : 'Explore stock returns, sized by dated ETF holdings weights.')), /*#__PURE__*/React.createElement("div", {
    className: "ph-actions"
  }, /*#__PURE__*/React.createElement("label", null, "Universe", /*#__PURE__*/React.createElement("select", {
    "aria-label": "Heat map universe",
    value: universe,
    onChange: e => {
      setUniverse(e.target.value);
      setNotice('');
    }
  }, /*#__PURE__*/React.createElement("option", {
    value: "portfolio"
  }, "My portfolio"), /*#__PURE__*/React.createElement("option", {
    value: "spx"
  }, "SPX \u2014 S&P 500"), /*#__PURE__*/React.createElement("option", {
    value: "nasdaq"
  }, "Nasdaq \u2014 Nasdaq-100"), /*#__PURE__*/React.createElement("option", {
    value: "rlv"
  }, "RLV \u2014 Russell 1000 Value"), /*#__PURE__*/React.createElement("option", {
    value: "rlg"
  }, "RLG \u2014 Russell 1000 Growth"))), universe === 'portfolio' && /*#__PURE__*/React.createElement("button", {
    onClick: () => {
      setEditing(!editing);
      setDraft(saved || blank());
      setNotice('');
    },
    disabled: !saved
  }, editing ? 'Close editor' : 'Edit holdings'), /*#__PURE__*/React.createElement("button", {
    onClick: () => setRefresh(r => r + 1),
    disabled: loading || !rows.length
  }, loading ? 'Loading prices…' : 'Refresh prices'))), error && /*#__PURE__*/React.createElement("div", {
    className: "ph-error",
    role: "alert"
  }, error, " ", !saved && /*#__PURE__*/React.createElement("button", {
    onClick: load
  }, "Retry loading holdings")), notice && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, notice), universe === 'portfolio' && editing && /*#__PURE__*/React.createElement("section", {
    className: "ph-editor"
  }, /*#__PURE__*/React.createElement("h2", null, "Define your portfolio"), /*#__PURE__*/React.createElement("p", null, "Enter actual position weights: 5 means 5%. Negative weights represent shorts. Use market-data tickers (for example BRK-B). Research coverage is not automatically treated as holdings."), /*#__PURE__*/React.createElement("fieldset", {
    disabled: saving
  }, /*#__PURE__*/React.createElement("div", {
    className: "ph-actions"
  }, /*#__PURE__*/React.createElement("label", null, "Portfolio name", /*#__PURE__*/React.createElement("input", {
    value: draft.name,
    onChange: e => setDraft({
      ...draft,
      name: e.target.value
    })
  })), /*#__PURE__*/React.createElement("label", null, "Holdings as of", /*#__PURE__*/React.createElement("input", {
    type: "date",
    value: draft.asOf,
    onChange: e => setDraft({
      ...draft,
      asOf: e.target.value
    })
  }))), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Import CSV or paste from a spreadsheet"), /*#__PURE__*/React.createElement("p", null, "Headers: ticker, weight, sector, company. Sector and company are optional. Import replaces the draft rows; saved holdings change only when you save."), /*#__PURE__*/React.createElement("button", {
    onClick: () => download('charlie-holdings-template.csv', 'ticker,weight,sector,company\n')
  }, "Download blank CSV template"), /*#__PURE__*/React.createElement("label", null, "Choose CSV", /*#__PURE__*/React.createElement("input", {
    type: "file",
    accept: ".csv,.tsv,.txt",
    onChange: async e => {
      var f = e.target.files[0];
      if (f) {
        if (f.size > 1000000) {
          setError('Use a CSV under 1 MB.');
          return;
        }
        importText(await f.text());
      }
      e.target.value = '';
    }
  })), /*#__PURE__*/React.createElement("textarea", {
    "aria-label": "Paste holdings CSV",
    placeholder: 'ticker,weight,sector,company',
    value: paste,
    onChange: e => setPaste(e.target.value)
  }), /*#__PURE__*/React.createElement("button", {
    onClick: () => importText(paste)
  }, "Import pasted rows")), /*#__PURE__*/React.createElement("div", {
    className: "ph-edit-rows"
  }, draft.holdings.map((r, i) => /*#__PURE__*/React.createElement("div", {
    className: "ph-edit-row",
    key: i
  }, /*#__PURE__*/React.createElement("label", null, "Ticker", /*#__PURE__*/React.createElement("input", {
    value: r.ticker,
    onChange: e => edit(i, 'ticker', e.target.value.toUpperCase())
  })), /*#__PURE__*/React.createElement("label", null, "Weight %", /*#__PURE__*/React.createElement("input", {
    type: "number",
    step: "any",
    value: r.weight,
    onChange: e => edit(i, 'weight', e.target.value)
  })), /*#__PURE__*/React.createElement("label", null, "Sector", /*#__PURE__*/React.createElement("input", {
    value: r.sector,
    onChange: e => edit(i, 'sector', e.target.value)
  })), /*#__PURE__*/React.createElement("label", null, "Company", /*#__PURE__*/React.createElement("input", {
    value: r.company,
    onChange: e => edit(i, 'company', e.target.value)
  })), /*#__PURE__*/React.createElement("button", {
    "aria-label": 'Remove ' + (r.ticker || 'holding'),
    onClick: () => setDraft(d => ({
      ...d,
      holdings: d.holdings.filter((_, j) => j !== i)
    }))
  }, "Remove")))), /*#__PURE__*/React.createElement("div", {
    className: "ph-actions"
  }, /*#__PURE__*/React.createElement("button", {
    disabled: draft.holdings.length >= 100,
    onClick: () => setDraft(d => ({
      ...d,
      holdings: [...d.holdings, {
        ticker: '',
        weight: '',
        sector: '',
        company: ''
      }]
    }))
  }, "+ Add holding"), /*#__PURE__*/React.createElement("button", {
    className: "ph-primary",
    disabled: !draft.holdings.length,
    onClick: save
  }, saving ? 'Saving…' : 'Save holdings'), /*#__PURE__*/React.createElement("button", {
    onClick: load
  }, "Reload saved snapshot")))), indexLoading && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Loading index holdings\u2026"), indexError && /*#__PURE__*/React.createElement("div", {
    className: "ph-error",
    role: "alert"
  }, indexError, " ", /*#__PURE__*/React.createElement("button", {
    onClick: () => setIndexRetry(r => r + 1)
  }, "Retry loading market")), universe !== 'portfolio' && active && /*#__PURE__*/React.createElement("div", {
    className: "ph-source"
  }, /*#__PURE__*/React.createElement("strong", null, active.proxy, " holdings proxy \xB7 ", active.asOf), /*#__PURE__*/React.createElement("p", null, active.basis, " ", /*#__PURE__*/React.createElement("a", {
    href: active.sourceUrl,
    target: "_blank",
    rel: "noreferrer"
  }, "Issuer holdings"), active.excludedEquities > 0 ? ` · ${active.excludedEquities} equity rows could not be mapped.` : ''), active.stale && /*#__PURE__*/React.createElement("p", {
    className: "ph-warning"
  }, "The issuer holdings date is over a week old.")), !rows.length ? universe === 'portfolio' ? /*#__PURE__*/React.createElement("section", {
    className: "ph-empty"
  }, /*#__PURE__*/React.createElement("h2", null, "A map of what you actually own"), /*#__PURE__*/React.createElement("p", null, "Add your tickers and portfolio weights above to create your heat map. Holdings are stored in Charlie and shared across desktop and mobile.")) : null : /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    className: "ph-stats"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("span", null, "POSITIONS"), /*#__PURE__*/React.createElement("strong", null, rows.length)), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("span", null, universe === 'portfolio' ? 'GROSS EXPOSURE' : 'ETF EQUITY WEIGHT'), /*#__PURE__*/React.createElement("strong", null, number(gross), "%")), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("span", null, "RETURN COVERAGE"), /*#__PURE__*/React.createElement("strong", null, loading ? '…' : number(gross ? covered / gross * 100 : 0) + '%')), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("span", null, "HOLDINGS AS OF"), /*#__PURE__*/React.createElement("strong", null, active.asOf))), universe === 'portfolio' && Math.floor((Date.now() - new Date(active.asOf + 'T12:00:00').getTime()) / 86400000) > 30 && /*#__PURE__*/React.createElement("p", {
    className: "ph-warning"
  }, "This holdings snapshot is over 30 days old. Update weights to keep tile sizes representative."), /*#__PURE__*/React.createElement("div", {
    className: "ph-toolbar"
  }, /*#__PURE__*/React.createElement("div", {
    className: "ph-periods",
    "aria-label": "Return period"
  }, periods.map(([id, label]) => /*#__PURE__*/React.createElement("button", {
    key: id,
    "aria-pressed": period === id,
    onClick: () => setPeriod(id)
  }, label))), /*#__PURE__*/React.createElement("div", {
    className: "ph-actions"
  }, /*#__PURE__*/React.createElement("label", {
    className: "ph-search"
  }, "Find a holding", /*#__PURE__*/React.createElement("input", {
    placeholder: "Ticker or company",
    value: filter,
    onChange: e => setFilter(e.target.value)
  })), /*#__PURE__*/React.createElement("label", null, "Sector", /*#__PURE__*/React.createElement("select", {
    "aria-label": "Sector",
    value: sector,
    onChange: e => setSector(e.target.value)
  }, /*#__PURE__*/React.createElement("option", null, "All sectors"), [...new Set(rows.map(r => r.sector))].sort().map(s => /*#__PURE__*/React.createElement("option", {
    key: s
  }, s)))), /*#__PURE__*/React.createElement("label", null, "View", /*#__PURE__*/React.createElement("select", {
    "aria-label": "View",
    value: view,
    onChange: e => setView(e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: "map"
  }, "Heat map"), /*#__PURE__*/React.createElement("option", {
    value: "list"
  }, "Holdings list"))))), /*#__PURE__*/React.createElement("div", {
    className: "ph-options"
  }, /*#__PURE__*/React.createElement("label", null, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: group,
    onChange: e => setGroup(e.target.checked)
  }), " Group by sector"), /*#__PURE__*/React.createElement("label", null, /*#__PURE__*/React.createElement("input", {
    type: "checkbox",
    checked: accessible,
    onChange: e => setAccessible(e.target.checked)
  }), " Blue / orange colors"), /*#__PURE__*/React.createElement("span", null, "Tile area = absolute weight \xB7 color = stock return")), /*#__PURE__*/React.createElement("div", {
    "aria-live": "polite",
    className: "ph-status"
  }, loading ? `Loading daily prices · ${progress} / ${rows.length} holdings checked…` : market ? `${market.provider} · fetched ${new Date(market.fetchedAt).toLocaleString()}` : 'Prices not loaded.'), chosen && /*#__PURE__*/React.createElement("aside", {
    className: "ph-selected"
  }, /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("strong", null, chosen.ticker, " \xB7 ", chosen.company), /*#__PURE__*/React.createElement("p", null, number(chosen.weight), "% weight", chosen.weight < 0 ? ' · Short position' : '', " \xB7 ", chosen.sector)), /*#__PURE__*/React.createElement("div", null, /*#__PURE__*/React.createElement("strong", null, formatReturn(chosen.changePct)), /*#__PURE__*/React.createElement("p", null, chosen.baselineDate ? `${chosen.baselineDate} → ${chosen.asOf}` : chosen.issue || 'Waiting for prices', chosen.stale ? ' · Stale quote' : '')), /*#__PURE__*/React.createElement("button", {
    onClick: () => onOpen(chosen.ticker)
  }, "Open company"), /*#__PURE__*/React.createElement("button", {
    "aria-label": "Close holding details",
    onClick: () => setSelected(null)
  }, "\xD7")), !visible.length ? /*#__PURE__*/React.createElement("p", {
    className: "ph-empty"
  }, "No holdings match these filters.") : view === 'map' ? /*#__PURE__*/React.createElement("div", {
    className: "ph-map",
    ref: mapRef,
    style: {
      height: Math.max(440, Math.min(660, width * .58))
    }
  }, groups.map(g => /*#__PURE__*/React.createElement(React.Fragment, {
    key: g.name
  }, /*#__PURE__*/React.createElement("div", {
    className: "ph-sector",
    style: {
      left: g.x,
      top: g.y,
      width: g.width,
      height: 24
    },
    title: g.name
  }, g.name), g.tiles.map(t => /*#__PURE__*/React.createElement("button", {
    key: t.ticker,
    className: "ph-tile",
    style: {
      left: t.x,
      top: t.y,
      width: t.width,
      height: t.height,
      background: tileColor(t.changePct, accessible)
    },
    "aria-label": `${t.ticker}, weight ${t.weight}%, return ${formatReturn(t.changePct)}${t.stale ? ', stale quote' : ''}`,
    title: `${t.company} (${t.ticker})\nWeight: ${t.weight}%\nReturn: ${formatReturn(t.changePct)}\n${t.asOf || t.issue || 'Waiting for prices'}`,
    onClick: () => setSelected(t.ticker)
  }, t.width > 47 && t.height > 35 && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("strong", {
    style: {
      fontSize: Math.max(12, Math.min(42, t.width / 5, t.height / 4))
    }
  }, t.ticker), t.height > 55 && /*#__PURE__*/React.createElement("span", {
    style: {
      fontSize: Math.max(10, Math.min(22, t.width / 7))
    }
  }, formatReturn(t.changePct)), t.height > 100 && t.width > 110 && /*#__PURE__*/React.createElement("small", null, number(t.weight), "% weight", t.weight < 0 ? ' · short' : '', t.stale ? ' · stale' : ''))))))) : /*#__PURE__*/React.createElement("div", {
    className: "ph-table-wrap"
  }, /*#__PURE__*/React.createElement("table", null, /*#__PURE__*/React.createElement("thead", null, /*#__PURE__*/React.createElement("tr", null, /*#__PURE__*/React.createElement("th", null, "Holding"), /*#__PURE__*/React.createElement("th", null, "Weight"), /*#__PURE__*/React.createElement("th", null, "Return"), /*#__PURE__*/React.createElement("th", null, "Price dates"))), /*#__PURE__*/React.createElement("tbody", null, visible.slice().sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight)).map(r => /*#__PURE__*/React.createElement("tr", {
    key: r.ticker
  }, /*#__PURE__*/React.createElement("td", null, /*#__PURE__*/React.createElement("button", {
    onClick: () => setSelected(r.ticker)
  }, /*#__PURE__*/React.createElement("b", null, r.ticker)), /*#__PURE__*/React.createElement("small", null, r.company, " \xB7 ", r.sector)), /*#__PURE__*/React.createElement("td", null, number(r.weight), "%"), /*#__PURE__*/React.createElement("td", null, /*#__PURE__*/React.createElement("span", {
    className: "ph-return",
    style: {
      background: tileColor(r.changePct, accessible)
    }
  }, formatReturn(r.changePct))), /*#__PURE__*/React.createElement("td", null, r.baselineDate ? `${r.baselineDate} → ${r.asOf}` : r.issue || 'Waiting', r.stale && /*#__PURE__*/React.createElement("small", null, "Stale quote"))))))), /*#__PURE__*/React.createElement("footer", {
    className: "ph-footer"
  }, /*#__PURE__*/React.createElement("div", {
    className: "ph-legend"
  }, [-3, -2, -1, 0, 1, 2, 3].map(v => /*#__PURE__*/React.createElement("span", {
    key: v,
    style: {
      background: tileColor(v, accessible)
    }
  }, v === -3 ? '≤ ' : v === 3 ? '≥ ' : '', v > 0 ? '+' : '', v, "%")), /*#__PURE__*/React.createElement("span", {
    style: {
      background: tileColor(null)
    }
  }, "No data")), /*#__PURE__*/React.createElement("p", null, "Daily dividend- and split-adjusted returns, potentially delayed and cached for up to 15 minutes; no extended-hours feed. 1 day compares the latest two available daily observations. Longer periods use the last observation on or before the calendar boundary (1 month = 30 days). Open a tile for its exact dates."), /*#__PURE__*/React.createElement("p", null, "Areas are proportional to the displayed holdings\u2019 absolute weights. Cash or unlisted exposure is not inferred. Colors show the stock\u2019s return even for shorts. This is a holdings snapshot, not portfolio P&L or performance attribution."), /*#__PURE__*/React.createElement("button", {
    onClick: () => download(`charlie-${universe}-heatmap.csv`, [['ticker', 'weight', 'sector', 'company', 'return_pct', 'baseline_date', 'price_date'], ...visible.map(r => [r.ticker, r.weight, r.sector, r.company, r.changePct ?? '', r.baselineDate || '', r.asOf || ''])].map(r => r.map(csvCell).join(',')).join('\n'))
  }, "Export displayed holdings & returns"))));
}
var styles = `
.portfolio-map{flex:1;min-height:0;height:100%;overflow:auto;padding:28px clamp(16px,3vw,42px) 90px;color:var(--ink);font-family:var(--font-body);min-width:0}.portfolio-map *{box-sizing:border-box}.portfolio-map h1{font-family:var(--font-display);font-size:32px;line-height:1.15;margin:7px 0 10px}.portfolio-map h2{font-size:21px;margin-bottom:12px}.portfolio-map p{line-height:1.55;margin:8px 0;color:var(--muted)}.ph-eyebrow{font:10px var(--font-mono);letter-spacing:.18em}.ph-heading,.ph-actions,.ph-options,.ph-toolbar,.ph-selected{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.ph-heading{justify-content:space-between;margin-bottom:24px}.portfolio-map button,.portfolio-map select,.portfolio-map input,.portfolio-map textarea{font:inherit;border:1px solid var(--line);border-radius:5px;padding:9px 12px;background:var(--surface);color:var(--ink);max-width:100%}.portfolio-map button{cursor:pointer;min-height:40px}.portfolio-map button:disabled{opacity:.5;cursor:wait}.portfolio-map button:focus-visible{outline:3px solid var(--accent);outline-offset:2px;z-index:3}.portfolio-map label{display:flex;flex-direction:column;gap:5px;font-size:12px}.portfolio-map textarea{width:100%;min-height:100px}.portfolio-map .ph-primary{background:var(--accent);color:var(--on-accent)}.ph-source{border:1px solid var(--line);border-radius:6px;padding:14px;margin-bottom:20px}.ph-source p{font-size:12px}.ph-source a{color:var(--accent)}.ph-editor{border:1px solid var(--line);padding:22px;border-radius:8px;background:var(--surface);margin-bottom:24px}.ph-editor details{margin:20px 0}.ph-editor summary{cursor:pointer;font-weight:600;margin:12px 0}.ph-edit-row{display:grid;grid-template-columns:1fr .8fr 1.4fr 1.6fr auto;gap:10px;align-items:end;margin:12px 0}.ph-edit-row input{width:100%}.ph-stats{display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid var(--line);border-bottom:1px solid var(--line);margin:20px 0}.ph-stats>div{padding:18px 12px;border-right:1px solid var(--line)}.ph-stats span{font:10px var(--font-mono);color:var(--muted);letter-spacing:.1em;display:block}.ph-stats strong{font-size:25px;font-weight:500;display:block;margin-top:8px}.ph-toolbar{justify-content:space-between;align-items:end}.ph-periods{display:flex;gap:3px;flex-wrap:wrap}.ph-periods button{font-size:12px}.ph-periods button[aria-pressed=true]{background:var(--ink);color:var(--bg);border-color:var(--ink)}.ph-options{margin:18px 0;font-size:12px;color:var(--muted)}.ph-options label{flex-direction:row;align-items:center}.ph-options input{width:16px;height:16px}.ph-status{font-size:11px;color:var(--muted);margin:12px 0}.ph-map{position:relative;background:#11161c;border:2px solid #11161c;border-radius:6px;overflow:hidden;width:100%}.portfolio-map .ph-tile{position:absolute;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:2px;color:#fff;border:1px solid #11161c;border-radius:0;min-height:0;padding:1px;overflow:hidden;line-height:1.12;font-family:Arial,sans-serif}.ph-tile:hover{box-shadow:inset 0 0 0 2px #fff;z-index:2}.ph-tile small{font-size:11px;opacity:.85;margin-top:8px}.ph-sector{position:absolute;color:#e1e8ed;background:#11161c;font:600 12px Arial,sans-serif;padding:5px 7px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.ph-selected{border:1px solid var(--accent);background:var(--surface);padding:14px;margin:12px 0;border-radius:6px;justify-content:space-between}.ph-selected p{font-size:12px}.ph-legend{display:flex;flex-wrap:wrap;margin:16px 0;gap:2px}.ph-legend span{color:white;font:11px Arial,sans-serif;padding:7px 12px}.ph-footer p{font-size:12px;max-width:1000px}.ph-empty{padding:42px 18px;border:1px dashed var(--line);margin:20px 0}.ph-error,.ph-warning{padding:14px;border:1px solid var(--accent);margin:14px 0}.ph-table-wrap{overflow:auto}.portfolio-map table{width:100%;border-collapse:collapse;font-size:13px}.portfolio-map td,.portfolio-map th{padding:12px 8px;border-bottom:1px solid var(--line);text-align:left}.portfolio-map td small{display:block;color:var(--muted);margin-top:4px}.ph-return{display:inline-block;color:white;padding:7px 9px;border-radius:4px;white-space:nowrap}@media(max-width:700px){.portfolio-map{padding:18px 12px 80px}.ph-heading h1{font-size:27px}.ph-stats{grid-template-columns:repeat(2,1fr)}.ph-stats strong{font-size:22px}.ph-edit-row{grid-template-columns:1fr 1fr;border-bottom:1px solid var(--line);padding-bottom:15px}.ph-editor{padding:14px}.ph-toolbar,.ph-toolbar .ph-actions{width:100%}.ph-search{width:100%}.ph-stats>div{padding:12px}.portfolio-map td,.portfolio-map th{padding:10px 5px;font-size:12px}.portfolio-map th:last-child,.portfolio-map td:last-child{display:none}.ph-options{align-items:flex-start}.ph-selected{position:static}.ph-legend span{padding:7px 8px}}
`;