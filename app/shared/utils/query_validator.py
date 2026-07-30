"""Query validator utility to detect and block malicious search inputs."""
import re
import logging

logger = logging.getLogger("query_validator")

# SQL Injection regex patterns
SQL_INJECTION_PATTERN = re.compile(
    r"(?i)("
    r"\b(OR|AND)\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+"             # OR 1=1, AND 1=1
    r"|\b(OR|AND)\s+['\"]?\w+['\"]?\s*=\s*['\"]?\w+"           # OR 'a'='a'
    r"|\bUNION\s+(ALL\s+)?SELECT\b"                           # UNION SELECT
    r"|\b(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|EXEC|SHOW|CREATE)\b\s+.*\b(FROM|INTO|TABLE|DATABASE|SCHEMA)\b"
    r"|\bINFORMATION_SCHEMA\b"                                  # information_schema
    r"|\bBENCHMARK\s*\("                                       # BENCHMARK()
    r"|\bSLEEP\s*\("                                           # SLEEP()
    r"|--\s*$"                                                 # Trailing -- comment
    r"|--\s*-"                                                 # -- - comment
    r"|/\*.*?\*/"                                               # Block comment /* */
    r"|'\s*(OR|AND)\s+'"                                        # ' OR '
    r"|\"\s*(OR|AND)\s+\""                                     # " OR "
    r")"
)

# XSS & Script Injection regex patterns
XSS_PATTERN = re.compile(
    r"(?i)("
    r"<\s*script[^>]*>"                                        # <script>
    r"|<\s*/\s*script\s*>"                                     # </script>
    r"|javascript\s*:"                                         # javascript:
    r"|vbscript\s*:"                                           # vbscript:
    r"|onload\s*="                                             # onload=
    r"|onerror\s*="                                            # onerror=
    r"|onclick\s*="                                            # onclick=
    r"|onmouseover\s*="                                        # onmouseover=
    r"|<\s*iframe[^>]*>"                                       # <iframe>
    r"|<\s*object[^>]*>"                                       # <object>
    r"|<\s*embed[^>]*>"                                        # <embed>
    r"|eval\s*\("                                             # eval()
    r"|document\.cookie"                                       # document.cookie
    r")"
)

# Punctuation-only pattern
ONLY_PUNCTUATION_PATTERN = re.compile(r"^[^\w\s]+$", re.UNICODE)


def is_malicious_or_invalid_query(query: str) -> bool:
    """
    Returns True if the search query contains SQL Injection, XSS, or garbage attack payloads.
    """
    if not query:
        return True

    trimmed = query.strip()
    if not trimmed:
        return True

    # Length sanity check
    if len(trimmed) > 300:
        logger.warning(f"Query blocked: Exceeds 300 characters limit ({len(trimmed)} chars)")
        return True

    # Check punctuation-only string
    if ONLY_PUNCTUATION_PATTERN.match(trimmed):
        logger.warning(f"Query blocked: Pure punctuation/symbol payload '{trimmed}'")
        return True

    # Check SQL Injection
    if SQL_INJECTION_PATTERN.search(trimmed):
        logger.warning(f"Query blocked: SQL Injection payload detected in '{trimmed}'")
        return True

    # Check XSS / Script Injection
    if XSS_PATTERN.search(trimmed):
        logger.warning(f"Query blocked: XSS payload detected in '{trimmed}'")
        return True

    return False
