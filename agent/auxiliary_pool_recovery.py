"""Credential-pool recovery for auxiliary (side-LLM) calls.

Split out of ``auxiliary_client`` so the facade does not grow. Names the tests patch
(``load_pool``, the error predicates, cache eviction) are read from the facade at call time.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def recover_provider_pool_entry(provider: str, exc: Exception, *, failed_api_key: str = "") -> Optional[Any]:
    """Rotate/refresh the provider's pool after a failed auxiliary call; the entry now serving it, else ``None``.

    ``failed_api_key`` lets mark_exhausted_and_rotate identify the right pool entry even if another process
    already rotated (current() would be None). The entry is returned so a retry can be pinned to its key.
    """
    import agent.auxiliary_client as aux

    normalized = aux._normalize_aux_provider(provider)
    try:
        pool = aux.load_pool(normalized)
    except Exception as load_exc:  # health: allow BLE001 -- unchanged fail-open boundary: no pool means no recovery
        logger.debug("Auxiliary client: could not load pool for %s recovery: %s", normalized, load_exc)
        return None
    if not pool or not pool.has_credentials():
        return None
    status_code = getattr(exc, "status_code", None)

    def _rotate(fallback_status: int) -> Optional[Any]:
        error_context: Dict[str, Any] = {"message": str(exc)}
        if status_code is not None:
            error_context["status_code"] = status_code
        next_entry = pool.mark_exhausted_and_rotate(
            status_code=status_code if status_code is not None else fallback_status,
            error_context=error_context, api_key_hint=failed_api_key or None,
        )
        if next_entry is None:
            return None
        aux._evict_cached_clients(normalized)
        return next_entry

    if aux._is_auth_error(exc):
        refreshed = pool.try_refresh_current()
        if refreshed is not None:
            aux._evict_cached_clients(normalized)
            return refreshed
        return _rotate(401)
    if aux._is_payment_error(exc):
        return _rotate(402)
    if aux._is_rate_limit_error(exc) and not aux._is_overloaded_error(exc):
        return _rotate(429)
    return None
