"""Resolve the same shared credentials for already-authorized recovery."""
from provider_credentials import get

def anthropic_recovery_key(get_db):
    return get('anthropic', get_db)
