"""Explain missing editorial sections without changing research or accepting claims."""
from stock_evidence import report_fields, missing_value, valid_citations


def readiness(run, rendered):
    """Diagnose three core sections using the renderer's actual selection outcome.

    Returned paths are evidence identities, never claim bodies. Matching checks
    do not establish accuracy, investment materiality or investor acceptance.
    """
    fields = report_fields(run.get('state', {}).get('report', {}))
    valid = valid_citations(run)
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
        rows.append(dict(section=key, label=label, status=status, reason=reason,
                         savedPaths=present, supportedPaths=supported))
    return rows
