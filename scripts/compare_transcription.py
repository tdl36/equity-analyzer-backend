#!/usr/bin/env python3
"""Compare a Gemini transcript against gpt-4o-transcribe-diarize on one meeting.

Why this exists: Charlie's transcripts already carry "Speaker N" labels, but on
MCK the same id held both management ("I would first say there are generally
some pretty positive macro trends") and the analyst ("Do you feel as confident
about your retail capability as you do on specialty?"). A label that merges the
two roles is worse than no label, because every later stage -- the assessment's
evasion check, the credibility rating, the Q&A attribution -- depends on who
said a thing.

Switching the transcriber is the highest blast-radius change available: every
downstream check validates the note *against* the transcript, so a fidelity
regression would be invisible. This scores both candidates on the two axes that
decide it:

  1. Role separation -- does any speaker id hold both management voice and
     interviewer questions?
  2. Verbatim fidelity -- are disfluencies, bare digit-run dates and figures
     still there, or has the text been tidied?

Read-only. It never writes to the database, never touches the iCloud original,
and costs nothing until --run is passed.

Usage:
  # free: score an existing transcript
  python scripts/compare_transcription.py --baseline mck.txt

  # paid: transcribe the audio too, then compare (~$0.36 for a 45-minute file)
  python scripts/compare_transcription.py --baseline mck.txt \
      --audio "~/.../MCK Mgmt Meeting @ DB - 091626.m4a" --out mck-openai.txt --run
"""
import argparse
import os
import re
import sys

MODEL = 'gpt-4o-transcribe-diarize'
MAX_BYTES = 25 * 1024 * 1024           # OpenAI's documented per-file cap
PRICE_PER_MINUTE = 0.006               # published rate, 2026-09-21

TURN = re.compile(r'^\s*([A-Za-z0-9 _-]{1,24}?)\s*:\s*(.*)$')
DISFLUENCY = re.compile(r'\b(um+|uh+|er|you know|i mean|sort of|kind of)\b', re.I)
# Management speaking about its own company, with no question in the opening.
MANAGEMENT = re.compile(r"\bwe (have|are|were|think|don't|do|can|know|built|make|see|expect)\b", re.I)
QUESTION = re.compile(r'^(so |and |but |do you|how |what |can you|is there|when |why |could you)', re.I)
# Spoken effective dates arrive as bare digit runs: "101" is 10/1, "71" is 7/1.
DIGIT_DATE = re.compile(r'\b(11|41|71|91|101)\b')
FIGURE = re.compile(r'\b\d+(?:\.\d+)?\s*(?:%|percent|bps|basis points|billion|million)\b', re.I)


def parse_turns(text):
    """Split a transcript into (speaker, utterance). Unlabelled text is one turn."""
    turns, speaker, buffer = [], None, []
    for line in (text or '').splitlines():
        match = TURN.match(line)
        if match and not line.startswith(' '):
            if speaker is not None:
                turns.append((speaker, ' '.join(buffer).strip()))
            speaker, buffer = match.group(1).strip(), [match.group(2)]
        elif line.strip():
            buffer.append(line.strip())
    if speaker is not None:
        turns.append((speaker, ' '.join(buffer).strip()))
    if not turns and (text or '').strip():
        turns = [('(unlabelled)', text.strip())]
    return [(who, said) for who, said in turns if said]


def score(text, label):
    turns = parse_turns(text)
    speakers = sorted({who for who, _ in turns})
    per_speaker = {}
    for who, said in turns:
        entry = per_speaker.setdefault(who, {'chars': 0, 'management': 0, 'questions': 0})
        entry['chars'] += len(said)
        if MANAGEMENT.search(said) and '?' not in said[:80]:
            entry['management'] += 1
        if QUESTION.match(said) and '?' in said:
            entry['questions'] += 1
    # A speaker credited with both roles is the failure this test is looking for.
    collisions = [w for w, e in per_speaker.items() if e['management'] >= 2 and e['questions'] >= 1]
    chars = len(text or '')
    per_1k = lambda n: round(n / (chars / 1000), 2) if chars else 0.0
    return {
        'label': label,
        'characters': chars,
        'turns': len(turns),
        'speakers': len(speakers),
        'role_collisions': collisions,
        'disfluencies_per_1k': per_1k(len(DISFLUENCY.findall(text or ''))),
        'digit_dates': len(DIGIT_DATE.findall(text or '')),
        'figures': len(FIGURE.findall(text or '')),
        'turns_starting_lowercase': sum(1 for _, s in turns if s[:1].islower()),
        'turns_ending_mid_sentence': sum(1 for _, s in turns if s and s[-1] not in '.?!"\''),
        'fragment_turns': sum(1 for _, s in turns if len(s) < 15),
        'per_speaker': per_speaker,
    }


def transcribe(audio_path, out_path):
    import openai
    key = os.environ.get('OPENAI_API_KEY', '').strip()
    if not key:
        sys.exit('OPENAI_API_KEY is not set.')
    size = os.path.getsize(audio_path)
    if size > MAX_BYTES:
        sys.exit(f'{size / 1e6:.1f}MB exceeds the {MAX_BYTES / 1e6:.0f}MB cap. '
                 'Transcode to mono at a lower bitrate first, so the whole meeting '
                 'goes in ONE request -- speaker ids are only consistent within a request.')
    client = openai.OpenAI(api_key=key, timeout=1800)
    print(f'Transcribing {os.path.basename(audio_path)} ({size / 1e6:.1f}MB) with {MODEL}...')
    with open(audio_path, 'rb') as handle:
        result = client.audio.transcriptions.create(
            model=MODEL, file=handle,
            response_format='diarized_json', chunking_strategy='auto')
    lines, speaker, buffer = [], None, []
    for segment in getattr(result, 'segments', None) or []:
        who = (getattr(segment, 'speaker', '') or 'Unknown').strip()
        said = (getattr(segment, 'text', '') or '').strip()
        if not said:
            continue
        if who != speaker and buffer:
            lines.append(f'{speaker}: ' + ' '.join(buffer))
            buffer = []
        speaker, _ = who, buffer.append(said)
    if buffer:
        lines.append(f'{speaker}: ' + ' '.join(buffer))
    text = '\n\n'.join(lines).strip() or (getattr(result, 'text', '') or '')
    if out_path:
        with open(out_path, 'w') as handle:
            handle.write(text)
        print(f'Wrote {out_path} ({len(text):,} characters)')
    return text


def report(rows):
    fields = [('characters', 'characters'), ('turns', 'turns'), ('speakers', 'distinct speakers'),
              ('disfluencies_per_1k', 'disfluencies /1k chars'), ('digit_dates', 'bare digit dates'),
              ('figures', 'figures (%, bps, bn)'), ('turns_starting_lowercase', 'turns starting lowercase'),
              ('turns_ending_mid_sentence', 'turns ending mid-sentence'), ('fragment_turns', 'fragment turns')]
    width = max(len(name) for _, name in fields) + 2
    print('\n' + ' ' * width + ''.join(f'{row["label"]:>22}' for row in rows))
    for key, name in fields:
        print(f'{name:<{width}}' + ''.join(f'{row[key]:>22}' for row in rows))
    print('\nRole separation — a speaker holding BOTH management voice and interviewer questions:')
    for row in rows:
        verdict = ', '.join(row['role_collisions']) if row['role_collisions'] else 'none — roles are separated'
        print(f'  {row["label"]:<22} {verdict}')
        for who, entry in sorted(row['per_speaker'].items(), key=lambda kv: -kv[1]['chars'])[:5]:
            print(f'      {who:<12} {entry["chars"]:>7} chars   '
                  f'{entry["management"]:>2} management   {entry["questions"]:>2} questions')
    print('\nRead these together: fewer role collisions is the gain, and disfluencies,')
    print('bare digit dates and figures must not drop — those are fidelity, not noise.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--baseline', required=True, help='existing Gemini transcript (text file)')
    parser.add_argument('--audio', help='audio to transcribe with OpenAI for comparison')
    parser.add_argument('--out', help='where to write the OpenAI transcript')
    parser.add_argument('--run', action='store_true', help='actually call the paid API')
    args = parser.parse_args()

    baseline = open(os.path.expanduser(args.baseline)).read()
    rows = [score(baseline, 'Gemini (current)')]

    if args.audio:
        audio = os.path.expanduser(args.audio)
        if args.run:
            rows.append(score(transcribe(audio, args.out and os.path.expanduser(args.out)), MODEL))
        else:
            size = os.path.getsize(audio) / 1e6
            print(f'DRY RUN. {os.path.basename(audio)} is {size:.1f}MB; '
                  f'a 45-minute file costs about ${45 * PRICE_PER_MINUTE:.2f}. Pass --run to spend it.')
    report(rows)


if __name__ == '__main__':
    main()
