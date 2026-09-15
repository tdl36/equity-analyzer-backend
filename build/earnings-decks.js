import * as React from 'react';
export function EarningsDecks({
  api,
  activityId,
  available
}) {
  var [decks, setDecks] = React.useState([]),
    [deck, setDeck] = React.useState(null),
    [body, setBody] = React.useState(null);
  var [mode, setMode] = React.useState('full'),
    [theme, setTheme] = React.useState('paper'),
    [slideIndex, setSlideIndex] = React.useState(0);
  var [busy, setBusy] = React.useState(''),
    [error, setError] = React.useState(''),
    [message, setMessage] = React.useState(''),
    [history, setHistory] = React.useState([]);
  var lock = React.useRef(false),
    attempt = React.useRef(null),
    alive = React.useRef(true);
  var dirty = !!deck && JSON.stringify(body) !== JSON.stringify(deck.body);
  async function json(path, options = {}) {
    var r = await fetch(`${api}${path}`, {
      ...options,
      signal: AbortSignal.timeout(30000),
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      }
    });
    var data = await r.json();
    if (!r.ok) throw Error(data.error || 'The request failed. Try again.');
    return data;
  }
  var select = d => {
    setDeck(d);
    setBody(structuredClone(d.body));
    setSlideIndex(0);
    setHistory(d.history || []);
  };
  var reload = async () => {
    var d = await json(`/api/earnings/decks?activityId=${encodeURIComponent(activityId)}`);
    if (alive.current) setDecks(d.decks.sort((a, b) => b.body.createdAt.localeCompare(a.body.createdAt)));
    return d;
  };
  React.useEffect(() => {
    alive.current = true;
    reload().catch(e => {
      if (alive.current) setError(e.message);
    });
    return () => {
      alive.current = false;
    };
  }, [api, activityId]);
  async function run(label, action) {
    if (lock.current) return;
    lock.current = true;
    setBusy(label);
    setError('');
    setMessage('');
    try {
      await action();
    } catch (e) {
      if (alive.current) setError(e.name === 'TimeoutError' ? 'Response not confirmed. Reload saved decks before retrying. Your recap is unchanged.' : e.message);
    } finally {
      lock.current = false;
      if (alive.current) setBusy('');
    }
  }
  var build = () => run('Building deck…', async () => {
    var config = JSON.stringify({
      mode,
      theme,
      activityId
    });
    if (!attempt.current || attempt.current.config !== config) attempt.current = {
      config,
      id: crypto.randomUUID()
    };
    var d = await json('/api/earnings/decks', {
      method: 'POST',
      body: JSON.stringify({
        activityId,
        mode,
        theme,
        requestId: attempt.current.id
      })
    });
    if (!alive.current) return;
    select(d);
    attempt.current = null;
    setMessage('Deck saved. Review the wording, then export an editable PowerPoint.');
    await reload();
  });
  var save = () => run('Saving revision…', async () => {
    var d = await json(`/api/earnings/decks/${deck.id}`, {
      method: 'PUT',
      body: JSON.stringify({
        revision: deck.revision,
        body
      })
    });
    if (!alive.current) return;
    var position = slideIndex;
    select(d);
    setSlideIndex(position);
    setMessage('New revision saved. Earlier versions remain available.');
    await reload();
  });
  var open = (id, revision) => run('Loading deck…', async () => {
    var d = await json(`/api/earnings/decks/${id}${revision ? `?revision=${revision}` : ''}`);
    if (alive.current) select(d);
  });
  var download = () => run('Preparing PowerPoint…', async () => {
    var r = await fetch(`${api}/api/earnings/decks/${deck.id}/export?revision=${deck.revision}`, {
      signal: AbortSignal.timeout(60000)
    });
    if (!r.ok) {
      var e = await r.json();
      throw Error(e.error || 'Export failed');
    }
    var blob = await r.blob();
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = `${body.ticker}_earnings_review_v${deck.revision}.pptx`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    setMessage('PowerPoint downloaded. Text and shapes are editable.');
  });
  var copy = () => run('Copying…', async () => {
    await navigator.clipboard.writeText(body.slides.map(s => `${s.title}\n${s.items.map(t => `• ${t}`).join('\n')}`).join('\n\n'));
    setMessage('All slides copied.');
  });
  var edit = (field, value) => setBody(b => ({
    ...b,
    slides: b.slides.map((s, i) => i === slideIndex ? {
      ...s,
      [field]: value
    } : s)
  }));
  var slide = body?.slides[slideIndex];
  return /*#__PURE__*/React.createElement("section", {
    className: "earnings-decks",
    "aria-label": "Earnings presentation builder"
  }, /*#__PURE__*/React.createElement("header", null, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "RECAP \u2192 PRESENTATION"), /*#__PURE__*/React.createElement("h3", null, "Your earnings review, ready to present."), /*#__PURE__*/React.createElement("p", null, "Build editable slides from this saved recap. Full coverage keeps every section; the brief uses the first two paragraphs per section. Full section text stays in speaker notes.")), /*#__PURE__*/React.createElement("div", {
    className: "ed-controls"
  }, /*#__PURE__*/React.createElement("label", null, "Coverage", /*#__PURE__*/React.createElement("select", {
    value: mode,
    onChange: e => setMode(e.target.value),
    disabled: !!busy
  }, /*#__PURE__*/React.createElement("option", {
    value: "full"
  }, "Full recap"), /*#__PURE__*/React.createElement("option", {
    value: "brief"
  }, "Section brief"))), /*#__PURE__*/React.createElement("label", null, "Slide style", /*#__PURE__*/React.createElement("select", {
    value: theme,
    onChange: e => setTheme(e.target.value),
    disabled: !!busy
  }, /*#__PURE__*/React.createElement("option", {
    value: "paper"
  }, "Editorial paper"), /*#__PURE__*/React.createElement("option", {
    value: "midnight"
  }, "Midnight boardroom"), /*#__PURE__*/React.createElement("option", {
    value: "sage"
  }, "Sage research"))), /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    disabled: !available || !!busy || dirty,
    onClick: build
  }, busy === 'Building deck…' ? busy : 'Build earnings deck')), !available && /*#__PURE__*/React.createElement("p", null, "Available when this event has a completed, saved recap."), /*#__PURE__*/React.createElement("p", {
    className: "ed-scope"
  }, "Formats existing research; does not run a new source search or approve a thesis change. The source register identifies recap inputs, not verified support for every slide claim."), /*#__PURE__*/React.createElement("div", {
    className: "ed-controls"
  }, /*#__PURE__*/React.createElement("label", null, "Saved decks", /*#__PURE__*/React.createElement("select", {
    "aria-label": "Saved earnings decks",
    value: deck?.id || '',
    disabled: !!busy || dirty,
    onChange: e => e.target.value && open(e.target.value)
  }, /*#__PURE__*/React.createElement("option", {
    value: ""
  }, "Choose a saved deck"), decks.map(d => /*#__PURE__*/React.createElement("option", {
    key: d.id,
    value: d.id
  }, new Date(d.body.createdAt).toLocaleString(), " \xB7 ", d.body.mode === 'full' ? 'Full' : 'Brief', " \xB7 v", d.revision)))), /*#__PURE__*/React.createElement("button", {
    disabled: !!busy || dirty,
    onClick: () => run('Refreshing…', async () => {
      await reload();
      if (deck) select(await json(`/api/earnings/decks/${deck.id}`));
    })
  }, "Reload saved decks")), error && /*#__PURE__*/React.createElement("p", {
    className: "workspace-error",
    role: "alert"
  }, error), /*#__PURE__*/React.createElement("p", {
    role: "status",
    "aria-live": "polite"
  }, busy || message), body && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("div", {
    className: "ed-toolbar"
  }, /*#__PURE__*/React.createElement("strong", null, body.slides.length, " slides \xB7 version ", deck.revision, dirty ? ' · unsaved edits' : ''), /*#__PURE__*/React.createElement("div", {
    className: "ed-controls"
  }, /*#__PURE__*/React.createElement("button", {
    disabled: !!busy || !dirty,
    onClick: save
  }, "Save new version"), /*#__PURE__*/React.createElement("button", {
    disabled: !!busy || dirty,
    onClick: download
  }, "Download PowerPoint"), /*#__PURE__*/React.createElement("button", {
    disabled: !!busy,
    onClick: copy
  }, "Copy all slides"), dirty && /*#__PURE__*/React.createElement("button", {
    disabled: !!busy,
    onClick: () => setBody(structuredClone(deck.body))
  }, "Discard edits"))), dirty && /*#__PURE__*/React.createElement("p", null, "Save or discard your edits before exporting or opening another deck."), body.warnings.map((w, i) => /*#__PURE__*/React.createElement("p", {
    className: "ed-scope",
    key: i
  }, w)), /*#__PURE__*/React.createElement("div", {
    className: "ed-studio"
  }, /*#__PURE__*/React.createElement("nav", {
    "aria-label": "Slides"
  }, body.slides.map((s, i) => /*#__PURE__*/React.createElement("button", {
    key: s.id,
    "aria-current": i === slideIndex ? 'step' : undefined,
    onClick: () => setSlideIndex(i)
  }, /*#__PURE__*/React.createElement("small", null, String(i + 1).padStart(2, '0'), " / ", s.kind === 'sources' ? 'Sources' : s.edited ? 'Edited draft' : 'Recap draft'), /*#__PURE__*/React.createElement("span", null, s.title)))), /*#__PURE__*/React.createElement("div", {
    className: "ed-canvas-column"
  }, /*#__PURE__*/React.createElement("article", {
    className: `ed-preview ed-${body.theme}`,
    "aria-label": `Slide ${slideIndex + 1} preview`
  }, /*#__PURE__*/React.createElement("p", {
    className: "ed-kicker"
  }, "CHARLIE / ", body.ticker, " / EARNINGS REVIEW"), /*#__PURE__*/React.createElement("h4", null, slide.title), /*#__PURE__*/React.createElement("ol", null, slide.items.map((t, i) => /*#__PURE__*/React.createElement("li", {
    key: i
  }, t))), /*#__PURE__*/React.createElement("footer", null, "Review draft \xB7 source register in appendix ", /*#__PURE__*/React.createElement("span", null, slideIndex + 1, " / ", body.slides.length))), slide.kind !== 'sources' && /*#__PURE__*/React.createElement("details", {
    className: "ed-editor"
  }, /*#__PURE__*/React.createElement("summary", null, "Edit this slide"), /*#__PURE__*/React.createElement("label", null, "Title", /*#__PURE__*/React.createElement("input", {
    value: slide.title,
    maxLength: 180,
    disabled: !!busy,
    onChange: e => edit('title', e.target.value)
  })), slide.items.map((t, i) => /*#__PURE__*/React.createElement("label", {
    key: i
  }, "Point ", i + 1, /*#__PURE__*/React.createElement("textarea", {
    "aria-label": `Point ${i + 1}`,
    value: t,
    maxLength: 240,
    rows: 3,
    disabled: !!busy,
    onChange: e => edit('items', slide.items.map((v, j) => j === i ? e.target.value : v))
  }), /*#__PURE__*/React.createElement("small", null, t.length, " / 240 characters"))), /*#__PURE__*/React.createElement("p", null, "Edits remain your interpretation and are not independently verified.")), /*#__PURE__*/React.createElement("details", {
    className: "ed-editor"
  }, /*#__PURE__*/React.createElement("summary", null, "Original section and evidence scope"), /*#__PURE__*/React.createElement("p", null, body.scope), /*#__PURE__*/React.createElement("pre", null, slide.notes || 'No original section text recorded.'), /*#__PURE__*/React.createElement("small", null, "Frozen recap reference: ", body.recapHash.slice(0, 16))))), /*#__PURE__*/React.createElement("div", {
    className: "ed-controls"
  }, /*#__PURE__*/React.createElement("button", {
    disabled: !!busy || dirty,
    onClick: () => open(deck.id)
  }, "Load version history"), history.length > 0 && /*#__PURE__*/React.createElement("label", null, "Saved version", /*#__PURE__*/React.createElement("select", {
    value: deck.revision,
    disabled: !!busy || dirty,
    onChange: e => open(deck.id, e.target.value)
  }, history.map(h => /*#__PURE__*/React.createElement("option", {
    key: h.revision,
    value: h.revision
  }, "Version ", h.revision, " \xB7 ", h.createdAt)))))));
}