"""Exact recorded-answer edits; no research dispatch or acceptance."""
FIELDS='q.id,q.meeting_id,q.question,q.response_notes,q.status'


def read(cur, answer_id, meeting_id=None, lock=False):
    cur.execute(f'''SELECT {FIELDS} FROM mp_past_questions q
        JOIN mp_meetings m ON m.id=q.meeting_id AND m.company_id=q.company_id
        WHERE q.id=%s''' + (' AND q.meeting_id=%s' if meeting_id is not None else '') +
        (' FOR UPDATE OF q' if lock else ''),
        (answer_id,meeting_id) if meeting_id is not None else (answer_id,))
    row=cur.fetchone()
    return dict(row) if row else None


def save(get_db, answer_id, data):
    if not isinstance(data,dict) or set(data)!={'meetingId','question','responseNotes','status','expected'}:
        return {'error':'Reload the meeting before saving: an exact answer baseline is required.'},400
    expected=data['expected']
    if (type(data['meetingId']) is not int or data['meetingId']<1 or
        not isinstance(data['question'],str) or not data['question'] or
        not isinstance(data['responseNotes'],str) or not data['responseNotes'].strip() or len(data['responseNotes'])>10000 or
        data['status']!='answered' or not isinstance(expected,dict) or set(expected)!={'responseNotes','status'} or
        not (expected['responseNotes'] is None or isinstance(expected['responseNotes'],str)) or
        not isinstance(expected['status'],str)):
        return {'error':'A valid recorded answer and its saved baseline are required.'},400
    with get_db(commit=True) as (_,cur):
        row=read(cur,answer_id,data['meetingId'],lock=True)
        if not row or row['question']!=data['question']:
            return {'error':'The recorded question does not match this meeting.'},404
        proposed=(data['responseNotes'],data['status'])
        current=(row['response_notes'],row['status'])
        if current==proposed:
            return {'answer':row,'replayed':True},200
        if current!=(expected['responseNotes'],expected['status']):
            return {'error':'This answer changed elsewhere. Review the saved answer before replacing it.','answer':row},409
        cur.execute('UPDATE mp_past_questions SET response_notes=%s,status=%s WHERE id=%s',(*proposed,answer_id))
        return {'answer':{**row,'response_notes':data['responseNotes'],'status':data['status']},'replayed':False},200
