"""Explain missing editorial sections without changing research or accepting claims."""
from onepager_edition_evidence import render_evidence, pending_label
from stock_evidence import report_fields, missing_value, valid_citations


def readiness(run, rendered, *, preview=None):
    """Diagnose three core sections using the renderer's actual selection outcome.

    Returned paths are evidence identities, never claim bodies. Matching checks
    do not establish accuracy, investment materiality or investor acceptance.
    """
    run, valid = render_evidence(run, preview)
    fields = report_fields(run.get('state', {}).get('report', {}))
    rows = []
    specs = [('thesis', 'Investment thesis', '/summary/investment_thesis/'),
             ('debate', 'Paired market debate', '/debates/'),
             ('view', 'Investment view', '/summary/earnings_direction')]
    debate_keys = {'market_view', 'evidence_for_market', 'alternative_view', 'evidence_for_alternative'}
    for key, label, prefix in specs:
        def relevant(path):
            if key == 'view':
                return path == prefix
            if not path.startswith(prefix):
                return False
            if key == 'debate':
                parts = path.split('/')
                return len(parts) == 4 and parts[2].isdigit() and parts[3] in debate_keys
            return True
        present = sorted(p for p, value in fields.items() if relevant(p) and not missing_value(value))
        supported = sorted(p for p in valid if relevant(p))
        if rendered.get(key):
            status, reason = 'rendered', 'Selected evidence appears on the page; accuracy and materiality still require review.'
        elif not present:
            status, reason = 'absent', 'No non-empty claim is saved for this section. Prepare source-checked evidence for review.'
        elif not supported:
            status, reason = 'unmatched', 'Saved claims do not pass the evidence identity and matching checks. Review their source passages.'
        elif key == 'debate':
            pairs = []
            for i in {p.split('/')[2] for p in supported}:
                root = '/debates/' + i + '/'
                if any(root+k in valid for k in ('market_view', 'evidence_for_market')) and any(root+k in valid for k in ('alternative_view', 'evidence_for_alternative')):
                    pairs.append(i)
            if not pairs:
                status, reason = 'incomplete_pair', 'No same-topic row has both supported sides. Do not combine different debate rows.'
            elif not any(str(i) in pairs for i in range(20)):
                status, reason = 'outside_selection', 'Supported pairs are outside the renderer’s first 20 debate rows. Review editorial selection.'
            else:
                status, reason = 'over_budget', 'Supported paired evidence exceeds the page limits. Prepare a qualified shorter version for review.'
        else:
            status, reason = 'over_budget', 'Supported evidence exceeds the page limits. Prepare a qualified shorter version for review.'
        row = dict(section=key, label=label, status=status, reason=reason, savedPaths=present, supportedPaths=supported)
        if preview:
            row['pendingPaths'] = [p for p in supported if pending_label(valid[p])]
            row['supportedPaths'] = [p for p in supported if not pending_label(valid[p])]
            row['reason'] = reason.replace('supported', 'evidence-matched').replace('Supported', 'Evidence-matched') + ' Pending proposals are not accepted; passage matching does not establish meaning.'
        rows.append(row)
    return rows


def scenario_readiness(run, *, preview=None):
    """Expose bounded per-row repair paths, never rejected financial statements.

    This is evidence readiness, independent of page selection or scenario labels.
    Financial fields never borrow support from another case.
    """
    from scenario_arithmetic import check
    run, valid = render_evidence(run, preview)
    report = run.get('state', {}).get('report', {})
    raw = report.get('scenarios', [])
    entries = list(enumerate(raw)) if isinstance(raw, list) else list(raw.items()) if isinstance(raw, dict) else []
    entries = [(str(i), row) for i, row in entries if str(i).isdigit() and isinstance(row, dict)]
    fields = report_fields(report)
    specs = [('business_conditions', 'Operating assumptions'), ('revenue_growth', 'Growth'),
             ('operating_margin', 'Margin'), ('eps_or_fcf', 'Earnings / cash flow'),
             ('valuation_multiple', 'Valuation multiple'), ('implied_value', 'Implied value'),
             ('upside_downside', 'Dated price return'), ('major_risks', 'Failure conditions')]
    rows = []
    for index, _ in entries[:20]:
        root = '/scenarios/' + index + '/'
        checks = []
        for key, label in specs:
            path = root + key
            status = 'absent' if missing_value(fields.get(path)) else 'supported' if path in valid else 'unmatched'
            if path in valid and pending_label(valid[path]): status = 'pending'
            checks.append(dict(path=path, label=label, status=status))
        arithmetic = check({key: valid.get(root + key) for key, _ in specs})
        rows.append(dict(path=root.rstrip('/'), fields=checks, arithmetic=arithmetic))
    return dict(rows=rows, omittedRows=max(0, len(entries)-20))
