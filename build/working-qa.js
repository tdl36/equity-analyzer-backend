import React, { useEffect, useState } from 'react';
import { documentHtml } from './summary-lab-format.mjs';
export function WorkingQA({
  api,
  kind,
  parentId,
  renderHtml
}) {
  var [row, setRow] = useState(null),
    [error, setError] = useState(''),
    [notice, setNotice] = useState(''),
    [busy, setBusy] = useState(false),
    [loaded, setLoaded] = useState(false);
  var base = `${api}/api/summary-lab`,
    path = `${base}/working-qa/${kind}/${encodeURIComponent(parentId)}`;
  var active = row && ['queued', 'running'].includes(row.status),
    text = row?.state?.sections?.working_qa || '',
    done = row?.status === 'complete';
  var stalled = active && row.updated_at && Date.now() - Date.parse(row.updated_at) > 15 * 60 * 1000;
  useEffect(() => {
    var live = true,
      timer;
    setRow(null);
    setError('');
    setNotice('');
    setLoaded(false);
    async function load() {
      try {
        var r = await fetch(path);
        var d = await r.json();
        if (!r.ok) throw Error(d.error || 'Could not load Working Notes.');
        if (live) {
          setRow(d.experiment);
          setLoaded(true);
          setError('');
        }
      } catch (e) {
        if (live) setError(e.message);
      } finally {
        if (live) timer = setTimeout(load, 12000);
      }
    }
    load();
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [path]);
  async function act(action) {
    setBusy(true);
    setError('');
    setNotice('');
    try {
      if (action === 'copy') {
        await navigator.clipboard.writeText(new DOMParser().parseFromString(documentHtml(text, renderHtml).replace(/<\/(p|li)>/g, '</$1>\n'), 'text/html').body.textContent || '');
        setNotice('Copied.');
        return;
      }
      var url = action === 'generate' ? path : `${base}/${row.id}/${action === 'save' ? 'save-to-icloud' : action}`;
      var r = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(action === 'save' ? {
          section: 'working_qa'
        } : {})
      });
      var d = await r.json();
      if (!r.ok) throw Error(d.error || 'Request failed.');
      if (action === 'generate') setRow(d.experiment);
      if (action === 'save') setNotice('Word document queued for iCloud. Your Mac agent must be online to deliver it.');
      if (action === 'stop') setNotice('Stop requested. Saved work is preserved.');
    } catch (e) {
      setError(e.message + ' No automatic resubmission was made.');
    } finally {
      setBusy(false);
    }
  }
  async function download() {
    setBusy(true);
    setError('');
    try {
      var r = await fetch(`${base}/${row.id}/working-qa.docx`);
      if (!r.ok) throw Error('Word download is unavailable.');
      var url = URL.createObjectURL(await r.blob()),
        a = document.createElement('a');
      a.href = url;
      a.download = 'Q&A Working Notes.docx';
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  return /*#__PURE__*/React.createElement("details", {
    className: "rounded-xl border border-white/10 overflow-hidden",
    style: {
      margin: '16px 0',
      padding: 16
    },
    open: true
  }, /*#__PURE__*/React.createElement("summary", {
    style: {
      cursor: 'pointer',
      fontWeight: 600
    }
  }, "Q&A \xB7 Working Notes"), /*#__PURE__*/React.createElement("p", {
    style: {
      fontSize: 13,
      opacity: .8,
      margin: '10px 0'
    }
  }, "Short questions and compact answer bullets, alongside your complete notes."), !row && loaded && /*#__PURE__*/React.createElement(React.Fragment, null, /*#__PURE__*/React.createElement("p", {
    style: {
      fontSize: 13
    }
  }, "Generate this additional section from the saved original. Uses model credits for source review, drafting and checks; other sections stay unchanged."), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => act('generate'),
    style: {
      padding: 10
    }
  }, busy ? 'Starting…' : 'Generate Q&A Working Notes')), !loaded && !error && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, "Loading saved Working Notes\u2026"), active && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, row.state?.progress || 'Queued', " ", /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => act('stop')
  }, "Stop")), stalled && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, "No progress has been saved for over 15 minutes. The run may be interrupted. No automatic restart has been requested."), row && ['failed', 'cancelled'].includes(row.status) && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, row.error || 'Generation stopped.', " Saved stages are preserved. Review this run in Summary Lab before resuming; an interrupted model call may have incurred credits. ", /*#__PURE__*/React.createElement("a", {
    href: "#view=summarylab"
  }, "Open Summary Lab")), done && /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      flexWrap: 'wrap',
      gap: 12,
      margin: '12px 0'
    }
  }, /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => act('copy')
  }, "Copy"), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: download
  }, "Download Word"), /*#__PURE__*/React.createElement("button", {
    disabled: busy,
    onClick: () => act('save')
  }, "Save to iCloud")), text && /*#__PURE__*/React.createElement(React.Fragment, null, !done && /*#__PURE__*/React.createElement("p", null, "Draft in progress; source checks may still revise it."), /*#__PURE__*/React.createElement("div", {
    className: "summary-content",
    style: {
      background: '#fff',
      color: '#000',
      fontFamily: 'Calibri, Carlito, sans-serif',
      padding: 20,
      overflowWrap: 'anywhere'
    },
    dangerouslySetInnerHTML: {
      __html: documentHtml(text, renderHtml)
    }
  })), row && /*#__PURE__*/React.createElement("details", {
    style: {
      marginTop: 12
    }
  }, /*#__PURE__*/React.createElement("summary", null, "Private review appendix"), /*#__PURE__*/React.createElement("pre", {
    style: {
      whiteSpace: 'pre-wrap',
      overflowWrap: 'anywhere'
    }
  }, row.state?.finalReview || 'Review not yet complete.'), (row.state?.editorialFlags?.working_qa || []).map((f, i) => /*#__PURE__*/React.createElement("p", {
    key: i
  }, f))), notice && /*#__PURE__*/React.createElement("p", {
    role: "status"
  }, notice), error && /*#__PURE__*/React.createElement("p", {
    role: "alert"
  }, error));
}