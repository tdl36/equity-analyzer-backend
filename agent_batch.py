"""Bounded execution for research batches; no database or API dependencies."""
import re
import threading
from concurrent.futures import ThreadPoolExecutor

MAX_CONCURRENCY = 3
MAX_BATCH_SIZE = 12
_PROCESS_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENCY)

def validate_batch(data):
    values = data.get('tickers')
    if not isinstance(values, list) or not values:
        raise ValueError('Choose at least one ticker.')
    tickers = []
    for value in values:
        if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.^=-]{0,19}', value.strip()):
            raise ValueError('Use valid ticker symbols (up to 20 characters).')
        ticker = value.strip().upper()
        if ticker not in tickers:
            tickers.append(ticker)
    if len(tickers) > MAX_BATCH_SIZE:
        raise ValueError(f'Choose at most {MAX_BATCH_SIZE} companies per batch.')
    concurrency = data.get('concurrency', 1)
    if type(concurrency) is not int or not 1 <= concurrency <= MAX_CONCURRENCY:
        raise ValueError(f'Concurrency must be between 1 and {MAX_CONCURRENCY}.')
    return tickers, concurrency

def run_batch_jobs(jobs, run, concurrency=1, on_error=None):
    """Share a per-process cap across batches; one failure cannot stop siblings.

    This keeps the existing background-thread lifecycle. It is not a durable
    distributed queue: worker restart recovery remains a separate concern.
    """
    if type(concurrency) is not int or not 1 <= concurrency <= MAX_CONCURRENCY:
        raise ValueError('Invalid concurrency')
    def execute(job):
        with _PROCESS_SLOTS:
            try:
                run(job)
            except Exception as exc:
                if on_error:
                    on_error(job, exc)
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix='research-batch') as pool:
        list(pool.map(execute, jobs))
