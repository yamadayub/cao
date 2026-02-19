"""PII encryption service using Fernet (AES-128-CBC)."""

from typing import Optional

from cryptography.fernet import Fernet

from app.config import get_settings


def get_fernet() -> Optional[Fernet]:
    """Get Fernet instance from ENCRYPTION_KEY env var."""
    key = get_settings().encryption_key
    if not key:
        return None
    return Fernet(key.encode())


def encrypt_pii(plaintext: str) -> Optional[str]:
    """Encrypt a PII string. Returns None if key is not set or input is empty."""
    f = get_fernet()
    if not f or not plaintext:
        return None
    return f.encrypt(plaintext.encode()).decode()


def decrypt_pii(ciphertext: str) -> Optional[str]:
    """Decrypt a PII string. Returns None if key is not set or input is empty."""
    f = get_fernet()
    if not f or not ciphertext:
        return None
    return f.decrypt(ciphertext.encode()).decode()
