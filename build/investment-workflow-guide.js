import * as React from 'react';
import { workflowSteps, workflowGuideHtml, syntheticOperatingModel, syntheticExpected } from './investment-workflow-guide.mjs';
export function InvestmentWorkflowGuide({
  api,
  active,
  onNavigate
}) {
  var [busy, setBusy] = React.useState(false),
    [result, setResult] = React.useState('');
  var guide = React.useRef(null);
  var alive = React.useRef(true);
  React.useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  var check = async () => {
    setBusy(true);
    setResult('Checking isolated synthetic inputs…');
    try {
      var send = model => fetch(api + '/api/research/operating-model/preview', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(model),
        signal: AbortSignal.timeout(20000)
      });
      var response = await send(syntheticOperatingModel());
      var d = await response.json();
      if (!response.ok) throw Error(d.error || 'Calculator unavailable');
      for (var [key, value] of Object.entries(syntheticExpected)) if (d.model?.results?.base?.[key] !== value) throw Error('Unexpected synthetic result for ' + key + '. No case was saved.');
      var invalid = syntheticOperatingModel();
      invalid.scenarios.base.shares = '0';
      var rejected = await send(invalid);
      if (rejected.status !== 400) throw Error('Zero shares were not rejected as expected.');
      if (alive.current) setResult('Passed: revenue 1,210.00m → EBITDA 242.00m → EV 2,420.00m → equity 2,195.00m → 21.95 USD/share; price return 9.75%; reverse EBITDA 222.50m. Zero shares correctly rejected. No case saved or paid research launched.');
    } catch (e) {
      if (alive.current) setResult('Check failed: ' + e.message);
    } finally {
      if (alive.current) setBusy(false);
    }
  };
  var download = () => {
    var url = URL.createObjectURL(new Blob([workflowGuideHtml()], {
      type: 'text/html;charset=utf-8'
    }));
    var a = document.createElement('a');
    a.href = url;
    a.download = 'Charlie-ticker-to-case-workflow-T120.html';
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return /*#__PURE__*/React.createElement("details", {
    ref: guide,
    className: "investment-workflow-guide"
  }, /*#__PURE__*/React.createElement("style", null, `.investment-workflow-guide{font-family:Calibri,sans-serif;color:#000;background:#f8fafb;padding:20px;margin:20px 0;border:1px solid #bac5cc;border-radius:8px}.investment-workflow-guide *{color:#000!important;overflow-wrap:anywhere}.investment-workflow-guide summary{font-size:20px;font-weight:700;cursor:pointer}.investment-workflow-guide article{border-top:1px solid #ccd4d9;padding:18px 0}.investment-workflow-guide h3{font-size:22px;margin:10px 0}.investment-workflow-guide p{margin:12px 0;max-width:90ch}.investment-workflow-guide button{background:#fff!important;border:1px solid #8796a2;border-radius:5px;padding:10px 14px;margin:8px 8px 8px 0}.investment-workflow-guide button:disabled{opacity:.5}`), /*#__PURE__*/React.createElement("summary", null, "Ticker-to-case workflow & test guide"), /*#__PURE__*/React.createElement("p", null, "Start with a ticker, build research from permitted originals, review the case, and calculate explicit scenarios. Opening saved content is free of model charges. Research and evidence assessment are separate paid actions."), /*#__PURE__*/React.createElement("button", {
    onClick: download
  }, "Download the complete workflow guide"), /*#__PURE__*/React.createElement("article", null, /*#__PURE__*/React.createElement("h3", null, "Start with a free calculator check"), /*#__PURE__*/React.createElement("p", null, "This isolated synthetic example checks arithmetic and invalid-input handling. It saves no company data and does not test research quality, collection or model-provider access."), /*#__PURE__*/React.createElement("p", null, "Base revenue 1,000m; FY2025 \u2192 FY2027; CAGR 10%; margin 20%; multiple 10\xD7; net debt 200m; claims 50m; nonoperating assets 25m; diluted shares 100m; reference price 20 USD. All three scenarios use the same inputs for this check."), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: check
  }, busy ? 'Checking…' : 'Run isolated calculator check'), /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, result)), workflowSteps.map((s, i) => /*#__PURE__*/React.createElement("article", {
    key: s.title
  }, /*#__PURE__*/React.createElement("h3", null, i + 1, ". ", s.title), /*#__PURE__*/React.createElement("p", null, s.action), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("strong", null, "Expected:"), " ", s.check), /*#__PURE__*/React.createElement("p", null, /*#__PURE__*/React.createElement("strong", null, "If blocked / limits:"), " ", s.stop), active && /*#__PURE__*/React.createElement("button", {
    onClick: () => {
      if (guide.current) guide.current.open = false;
      onNavigate(s.tab);
    }
  }, "Open ", s.tab === 'model' ? 'Operating scenarios' : s.tab === 'case' ? 'Current thesis' : s.tab === 'research' ? 'Deep Research' : s.tab === 'signals' ? 'Case signals' : 'Snapshot', " for ", active))));
}