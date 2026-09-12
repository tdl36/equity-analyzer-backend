import * as React from 'react';
import { defaultRules, pressure, retention, falsification, variant, divergence } from './case-signals-model.mjs';
var today = () => new Date().toISOString().slice(0, 10);
var blank = () => ({
  observations: [],
  tests: [],
  variants: [],
  position: {
    conviction: 'unknown'
  }
});
var fmt = n => n == null ? '—' : Number(n).toLocaleString(undefined, {
  maximumFractionDigits: 2
});
export function CaseSignals({
  api,
  body,
  disabled,
  onChange,
  onSave,
  message
}) {
  var s = {
      ...blank(),
      ...body.signals
    },
    assumptions = body.assumptions || [];
  var [framework, setFramework] = React.useState(null),
    [error, setError] = React.useState('');
  React.useEffect(() => {
    var live = true;
    var load = async () => {
      try {
        var r = await fetch(api + '/api/research/investor-framework', {
          signal: AbortSignal.timeout(20000)
        });
        var d = await r.json();
        if (!r.ok) throw Error(d.error || 'Doctrine unavailable');
        if (live) {
          setFramework(d);
          setError('');
        }
      } catch (e) {
        if (live) setError(e.message);
      }
    };
    load();
    window.addEventListener('charlie-framework-saved', load);
    return () => {
      live = false;
      window.removeEventListener('charlie-framework-saved', load);
    };
  }, [api]);
  var rules = framework?.body?.decayRules || defaultRules;
  var pillars = pressure(assumptions, s.observations, rules),
    position = divergence(s.position, pillars);
  var update = next => onChange({
    ...body,
    signals: next
  });
  var rowEdit = (key, id, field, value) => update({
    ...s,
    [key]: s[key].map(r => r.id === id ? {
      ...r,
      [field]: value
    } : r)
  });
  var remove = (key, id) => update({
    ...s,
    [key]: s[key].filter(r => r.id !== id)
  });
  var add = (key, extra = {}) => update({
    ...s,
    [key]: [...s[key], {
      id: crypto.randomUUID(),
      assumptionId: assumptions[0]?.id || '',
      ...extra
    }]
  });
  var input = (label, value, change, type = 'text') => /*#__PURE__*/React.createElement("label", null, label, /*#__PURE__*/React.createElement("input", {
    type: type,
    step: type === 'number' ? 'any' : undefined,
    value: value ?? '',
    onChange: e => change(e.target.value)
  }));
  var select = (label, value, change, options) => /*#__PURE__*/React.createElement("label", null, label, /*#__PURE__*/React.createElement("select", {
    value: value || '',
    onChange: e => change(e.target.value)
  }, options.map(([v, t]) => /*#__PURE__*/React.createElement("option", {
    key: v,
    value: v
  }, t))));
  var editor = (key, r) => ({
    field: (k, label, type = 'text') => input(label, r[k], v => rowEdit(key, r.id, k, v), type),
    choice: (k, label, options) => select(label, r[k], v => rowEdit(key, r.id, k, v), options)
  });
  var assumptionSelect = (key, r) => select('Linked pillar', r.assumptionId, v => rowEdit(key, r.id, 'assumptionId', v), assumptions.map(a => [a.id, a.claim]));
  return /*#__PURE__*/React.createElement("section", {
    className: "case-signals",
    "aria-label": "Case signals"
  }, /*#__PURE__*/React.createElement("header", null, /*#__PURE__*/React.createElement("p", {
    className: "workspace-eyebrow"
  }, "CASE SIGNALS / EVIDENCE & JUDGMENT"), /*#__PURE__*/React.createElement("h2", null, "Where does the case need attention?"), /*#__PURE__*/React.createElement("p", null, "Transparent diagnostics from your classified evidence and dated inputs. Save these annotations as an investment-case revision. No trades or weight recommendations.")), /*#__PURE__*/React.createElement("nav", {
    className: "signals-jump",
    "aria-label": "Case signal sections"
  }, ['Pillar pressure', 'Evidence half-life', 'Falsification', 'Variant view', 'Position alignment'].map((label, i) => /*#__PURE__*/React.createElement("a", {
    href: '#case-signal-' + i,
    key: label,
    onClick: e => {
      e.preventDefault();
      document.getElementById('case-signal-' + i)?.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
      });
    }
  }, label))), !assumptions.length && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Add and save a thesis assumption first."), error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error, ". Decay calculations below use the labeled defaults until doctrine loads."), /*#__PURE__*/React.createElement("section", {
    className: "signals-section",
    id: "case-signal-0"
  }, /*#__PURE__*/React.createElement("h3", null, "1. Pillar pressure map"), /*#__PURE__*/React.createElement("p", null, "Each active observation contributes one evidence unit, adjusted for age. Counts are not independent corroboration or a probability. Repeated reports of the same development should be one observation."), /*#__PURE__*/React.createElement("div", {
    className: "signals-grid"
  }, pillars.map(p => /*#__PURE__*/React.createElement("article", {
    key: p.id,
    className: "signal-card"
  }, /*#__PURE__*/React.createElement("h4", null, p.claim), /*#__PURE__*/React.createElement("div", {
    className: "signal-totals"
  }, /*#__PURE__*/React.createElement("span", null, "Support ", /*#__PURE__*/React.createElement("strong", null, fmt(p.support))), /*#__PURE__*/React.createElement("span", null, "Challenge ", /*#__PURE__*/React.createElement("strong", null, fmt(p.challenge)))), /*#__PURE__*/React.createElement("div", {
    className: "signal-bar",
    "aria-label": `${fmt(p.support)} supporting and ${fmt(p.challenge)} challenging retained units`
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      width: `${p.support + p.challenge ? p.support / (p.support + p.challenge) * 100 : 0}%`
    }
  })), /*#__PURE__*/React.createElement("p", null, p.rows.length ? `${p.unresolved} active challenging observation(s). Aging does not resolve them.` : 'No observations assessed. Pressure is unknown.'), /*#__PURE__*/React.createElement("ul", null, p.targets.map(t => /*#__PURE__*/React.createElement("li", {
    key: t.type
  }, t.type, ": ", t.support, " supporting / ", t.challenge, " challenging")))))), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Assess evidence \xB7 ", s.observations.length, " observations"), /*#__PURE__*/React.createElement("fieldset", {
    disabled: disabled || !assumptions.length
  }, (body.evidenceLinks || []).length > 0 && /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Start from accepted source passages"), body.evidenceLinks.map((link, i) => /*#__PURE__*/React.createElement("button", {
    key: i,
    disabled: s.observations.length >= 200 || !assumptions.some(a => a.id === link.assumptionId),
    onClick: () => {
      var refs = link.evidence || [];
      add('observations', {
        assumptionId: link.assumptionId,
        statement: refs.map(e => e.excerpt).join('\n'),
        source: refs.map(e => e.source?.filename || e.sourceId).join('; '),
        asOf: '',
        targetType: 'pillar',
        targetLabel: '',
        direction: 'neutral',
        kind: 'event',
        status: 'active'
      });
    }
  }, "Assess accepted passage ", i + 1)), /*#__PURE__*/React.createElement("p", null, "Enter the source observation date and your assessment. A saved citation is not automatically supporting or challenging.")), s.observations.map((r, i) => {
    var e = editor('observations', r);
    return /*#__PURE__*/React.createElement("article", {
      key: r.id,
      className: "signal-editor"
    }, /*#__PURE__*/React.createElement("h4", null, "Observation ", i + 1), assumptionSelect('observations', r), /*#__PURE__*/React.createElement("div", {
      className: "signals-grid"
    }, e.choice('targetType', 'What is affected?', [['pillar', 'Pillar'], ['signpost', 'Signpost'], ['estimate', 'Estimate'], ['risk', 'Risk']]), e.field('targetLabel', 'Specific signpost, estimate or risk'), e.choice('direction', 'Your assessment', [['neutral', 'Neutral / unresolved'], ['support', 'Supports the case'], ['challenge', 'Challenges the case']]), e.choice('kind', 'Evidence lifetime', [['event', 'Time-sensitive event'], ['cyclical', 'Cyclical development'], ['structural', 'Structural evidence']]), e.field('asOf', 'Observation date', 'date'), e.choice('status', 'Status', [['active', 'Active'], ['retired', 'Resolved / superseded — retain history']])), /*#__PURE__*/React.createElement("label", null, "Source statement / assessment", /*#__PURE__*/React.createElement("textarea", {
      value: r.statement || '',
      onChange: ev => rowEdit('observations', r.id, 'statement', ev.target.value)
    })), e.field('source', 'Source reference · original, page / passage'), /*#__PURE__*/React.createElement("button", {
      onClick: () => remove('observations', r.id)
    }, "Remove observation from this version"));
  }), /*#__PURE__*/React.createElement("button", {
    disabled: s.observations.length >= 200,
    onClick: () => add('observations', {
      targetType: 'pillar',
      direction: 'neutral',
      kind: 'event',
      status: 'active',
      asOf: today()
    })
  }, "Add evidence assessment")))), /*#__PURE__*/React.createElement("section", {
    className: "signals-section",
    id: "case-signal-1"
  }, /*#__PURE__*/React.createElement("h3", null, "2. Evidence half-life"), /*#__PURE__*/React.createElement("p", null, framework?.body?.decayRules ? `Analyst Doctrine / framework v${framework.revision}` : 'Preview defaults · save explicit rules in My investment framework / Analyst Doctrine above', ": event ", rules.event, " days; cyclical ", rules.cyclical, " days; structural evidence persists until retired. Retention = \xBD ^ (age \xF7 half-life). This measures relevance under your policy, not truth or confidence."), /*#__PURE__*/React.createElement("ul", null, s.observations.map((r, i) => /*#__PURE__*/React.createElement("li", {
    key: r.id
  }, /*#__PURE__*/React.createElement("strong", null, "Observation ", i + 1), " \xB7 ", r.kind, " \xB7 ", r.status === 'retired' ? 'Retired' : `${fmt(retention(r, rules) * 100)}% retained`, " \xB7 ", r.statement?.slice(0, 160) || 'Draft assessment', !r.asOf ? ' · Observation date required' : '')))), /*#__PURE__*/React.createElement("section", {
    className: "signals-section",
    id: "case-signal-2"
  }, /*#__PURE__*/React.createElement("h3", null, "3. Falsification dashboard"), /*#__PURE__*/React.createElement("p", null, "State what would disprove each pillar. Distance is in the entered unit; it is not a forecast or standardized risk score. A stale observation does not establish a current threshold breach."), /*#__PURE__*/React.createElement("fieldset", {
    disabled: disabled || !assumptions.length
  }, s.tests.map(r => {
    var e = editor('tests', r),
      v = falsification(r);
    return /*#__PURE__*/React.createElement("article", {
      className: "signal-editor",
      key: r.id
    }, assumptionSelect('tests', r), e.field('question', 'What would prove this wrong?'), /*#__PURE__*/React.createElement("p", {
      className: "signal-result"
    }, /*#__PURE__*/React.createElement("strong", null, v.status), v.distance !== null && /*#__PURE__*/React.createElement(React.Fragment, null, " \xB7 ", fmt(Math.abs(v.distance)), " ", r.unit, " ", v.distance <= 0 ? 'beyond / at threshold' : 'remaining to threshold')), /*#__PURE__*/React.createElement("details", null, /*#__PURE__*/React.createElement("summary", null, "Observation, threshold and evidence needed"), /*#__PURE__*/React.createElement("div", {
      className: "signals-grid"
    }, e.field('metric', 'Metric'), e.field('unit', 'Unit · e.g. %, USD/share'), e.field('period', 'Comparable fiscal period / basis'), e.field('current', 'Current observed value', 'number'), e.choice('operator', 'Disprove if observation is', [['lte', 'At or below threshold'], ['gte', 'At or above threshold']]), e.field('threshold', 'Threshold', 'number'), e.field('asOf', 'Observation as of', 'date'), e.field('source', 'Original source reference')), e.field('needed', 'Evidence needed to resolve uncertainty'), /*#__PURE__*/React.createElement("button", {
      onClick: () => remove('tests', r.id)
    }, "Remove test from this version")));
  }), /*#__PURE__*/React.createElement("button", {
    disabled: s.tests.length >= 60,
    onClick: () => add('tests', {
      operator: 'lte'
    })
  }, "Add falsification test"))), /*#__PURE__*/React.createElement("section", {
    className: "signals-section",
    id: "case-signal-3"
  }, /*#__PURE__*/React.createElement("h3", null, "4. Variant-perception tracker"), /*#__PURE__*/React.createElement("p", null, "Separate a positive business view from a view the market may not reflect. Estimate gaps do not establish mispricing. Pricing status below is your explicit interpretation; broker baselines are labeled separately from consensus."), /*#__PURE__*/React.createElement("fieldset", {
    disabled: disabled || !assumptions.length
  }, s.variants.map(r => {
    var e = editor('variants', r),
      v = variant(r);
    return /*#__PURE__*/React.createElement("article", {
      className: "signal-editor",
      key: r.id
    }, assumptionSelect('variants', r), /*#__PURE__*/React.createElement("div", {
      className: "signals-grid"
    }, e.field('metric', 'Metric'), e.field('unit', 'Unit'), e.field('period', 'Same forecast period / accounting basis'), e.field('view', 'My estimate', 'number'), e.field('market', 'Stated market / broker estimate', 'number'), e.choice('baselineType', 'Baseline type', [['unknown', 'Not established'], ['consensus', 'Documented consensus aggregation'], ['broker', 'Single broker — not consensus'], ['implied', 'Explicit market-implied calculation']]), e.field('asOf', 'Baseline as of', 'date'), e.field('source', 'Baseline source / calculation reference'), e.choice('pricing', 'My pricing assessment', [['unknown', 'Not established'], ['reflected', 'Already reflected'], ['not_reflected', 'Not reflected — my interpretation']])), /*#__PURE__*/React.createElement("p", {
      className: "signal-result"
    }, v.status, v.delta !== null && /*#__PURE__*/React.createElement(React.Fragment, null, " \xB7 My estimate minus baseline: ", fmt(v.delta), " ", r.unit, v.percent !== null ? ` (${fmt(v.percent)}%)` : '')), /*#__PURE__*/React.createElement("label", null, "Why I think it is / is not reflected", /*#__PURE__*/React.createElement("textarea", {
      value: r.rationale || '',
      onChange: ev => rowEdit('variants', r.id, 'rationale', ev.target.value)
    })), /*#__PURE__*/React.createElement("button", {
      onClick: () => remove('variants', r.id)
    }, "Remove comparison from this version"));
  }), /*#__PURE__*/React.createElement("button", {
    disabled: s.variants.length >= 60,
    onClick: () => add('variants', {
      baselineType: 'unknown',
      pricing: 'unknown'
    })
  }, "Add variant comparison"))), /*#__PURE__*/React.createElement("section", {
    className: "signals-section",
    id: "case-signal-4"
  }, /*#__PURE__*/React.createElement("h3", null, "5. Position\u2013conviction divergence"), /*#__PURE__*/React.createElement("p", null, "User-entered position snapshot; no live holdings feed. Weights are percentages, active weight is percentage points. Holdings older than 30 days require refresh."), /*#__PURE__*/React.createElement("fieldset", {
    disabled: disabled
  }, /*#__PURE__*/React.createElement("div", {
    className: "signals-grid"
  }, [['portfolio', 'Portfolio / mandate', 'text'], ['benchmark', 'Benchmark', 'text'], ['weight', 'Portfolio weight %', 'number'], ['benchmarkWeight', 'Benchmark weight %', 'number'], ['asOf', 'Weights as of', 'date']].map(([k, l, t]) => /*#__PURE__*/React.createElement(React.Fragment, {
    key: k
  }, input(l, s.position[k], v => update({
    ...s,
    position: {
      ...s.position,
      [k]: v
    }
  }), t))), select('Case conviction · your assessment', s.position.conviction, v => update({
    ...s,
    position: {
      ...s.position,
      conviction: v
    }
  }), [['unknown', 'Not assessed'], ['low', 'Low'], ['medium', 'Medium'], ['high', 'High']])), /*#__PURE__*/React.createElement("p", null, "Active weight: ", /*#__PURE__*/React.createElement("strong", null, fmt(position.active), " pp")), /*#__PURE__*/React.createElement("ul", null, position.questions.map(q => /*#__PURE__*/React.createElement("li", {
    key: q
  }, q))), /*#__PURE__*/React.createElement("label", null, "Position rationale / constraints", /*#__PURE__*/React.createElement("textarea", {
    value: s.position.rationale || '',
    onChange: e => update({
      ...s,
      position: {
        ...s.position,
        rationale: e.target.value
      }
    })
  })))), /*#__PURE__*/React.createElement("div", {
    className: "signal-save"
  }, /*#__PURE__*/React.createElement("button", {
    className: "workspace-primary",
    disabled: disabled || !assumptions.length,
    onClick: () => onSave()
  }, "Save signals with investment case"), /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, message || 'Annotations remain a draft until saved. Previous versions are retained in case history.')));
}