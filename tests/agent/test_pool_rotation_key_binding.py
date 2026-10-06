"""A rate-limited pool entry's failure must stay on that entry, and a retry must leave it.

Two paths used to send a request with one pool entry's token while attributing the outcome to another:

* The main loop's per-request Anthropic credential refresh resolved the pool's FIRST available token, so a
  session bound to entry B silently sent entry A's token again once A's short bench lapsed; A's 429
  was then recorded on B.
* The auxiliary credential rung retried an ``auto`` route after rotating, but ``auto`` resolves the
  session's main key, so the retry reused the failed key and the keyless follow-up mark benched the
  innocent rotated-to entry.

Real ``CredentialPool`` against a temp ``HERMES_HOME``; only the HTTP errors are constructed.
"""

import json
import time
from types import SimpleNamespace

import pytest

KEY_A, KEY_B, KEY_C = "sk-ant-oat01-AAAA-test", "sk-ant-oat01-BBBB-test", "sk-ant-oat01-CCCC-test"
MAIN_RUNTIME = {"provider": "anthropic", "api_key": KEY_A, "base_url": "https://api.anthropic.com",
                "api_mode": "anthropic_messages", "model": "claude-sonnet-4-5"}


def _entry(entry_id, label, priority, token, **extra):
    return {"id": entry_id, "label": label, "auth_type": "oauth", "priority": priority,
            "source": "manual:hermes_pkce", "access_token": token, "refresh_token": "r-" + label,
            "expires_at_ms": int((time.time() + 86400) * 1000), **extra}


@pytest.fixture
def pool_home(tmp_path, monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("model:\n  provider: anthropic\n  default: claude-sonnet-4-5\n")
    # Keep the pool to the rows written here: no borrowed Claude Code login from the host.
    monkeypatch.setattr("agent.anthropic_credentials.read_claude_code_credentials", lambda: None)

    def write(entries):
        (tmp_path / "auth.json").write_text(json.dumps({"version": 1, "credential_pool": {"anthropic": entries}}))
    return write


def _status(label):
    from agent.credential_pool import load_pool
    entry = next(e for e in load_pool("anthropic").entries() if e.label == label)
    return entry.last_status, entry.last_error_reason


class _RateLimited(Exception):
    status_code = 429

    def __init__(self):
        super().__init__("Error code: 429 - rate_limit_error: This request would exceed your account's rate limit.")
        self.response = None


class TestMainLoopRefreshKeepsTheBoundEntry:
    def _agent(self, bound_id, token):
        import agent.client_lifecycle as lifecycle
        from agent.credential_pool import load_pool
        cls = lifecycle.ClientLifecycleMixin
        agent = cls.__new__(cls)
        agent.__dict__.update(
            api_mode="anthropic_messages", provider="anthropic", model="claude-sonnet-4-5",
            _anthropic_base_url="https://api.anthropic.com", _anthropic_api_key=token, api_key=token,
            _credential_pool=load_pool("anthropic"), _credential_pool_entry_id=bound_id,
            _is_anthropic_oauth=True, _anthropic_client=SimpleNamespace(close=lambda: None),
        )
        agent._build_direct_anthropic_client = lambda key, base: SimpleNamespace(close=lambda: None, key=key)
        return agent

    def test_lapsed_bench_on_a_higher_priority_entry_does_not_pull_the_session_back(self, pool_home):
        pool_home([
            _entry("aaaa01", "A", 0, KEY_A, last_status="exhausted", last_status_at=time.time() - 600,
                   last_error_code=429),
            _entry("bbbb02", "B", 1, KEY_B),
        ])
        agent = self._agent("bbbb02", KEY_B)

        agent._try_refresh_anthropic_client_credentials()

        assert agent._anthropic_api_key == KEY_B

    def test_bound_entry_token_refreshed_elsewhere_is_adopted(self, pool_home):
        pool_home([_entry("aaaa01", "A", 0, KEY_A), _entry("bbbb02", "B", 1, KEY_C)])
        agent = self._agent("bbbb02", KEY_B)  # B's token was rotated to KEY_C by another process

        assert agent._try_refresh_anthropic_client_credentials() is True
        assert agent._anthropic_api_key == KEY_C

    def test_unbound_session_still_uses_the_resolver(self, pool_home):
        pool_home([_entry("aaaa01", "A", 0, KEY_A), _entry("bbbb02", "B", 1, KEY_B)])
        agent = self._agent(None, KEY_B)

        agent._try_refresh_anthropic_client_credentials()

        assert agent._anthropic_api_key == KEY_A

    def test_failure_identity_is_the_key_the_request_carried(self, pool_home):
        from agent.agent_runtime_helpers import _failed_credential_identity
        pool_home([_entry("aaaa01", "A", 0, KEY_A), _entry("bbbb02", "B", 1, KEY_B)])
        agent = self._agent("bbbb02", KEY_B)
        agent._anthropic_api_key = KEY_A  # what the request client was built from

        hint, _credential_id = _failed_credential_identity(agent, agent._credential_pool)

        assert hint == KEY_A


class TestAuxiliaryRotationRetry:
    def _route(self, aux):
        client, model = aux._get_cached_client("auto", None, main_runtime=MAIN_RUNTIME)
        assert client.api_key == KEY_A
        return aux._LadderRoute(
            client=client, task="approval", tag="", async_mode=False, base_info=None,
            resolved_provider="auto", resolved_model=None, resolved_base_url=None, resolved_api_key=None,
            resolved_api_mode=None, final_model=model, main_runtime=MAIN_RUNTIME, route_info=None, timeout=30,
        ), model

    def _retry_client(self, aux, step, model):
        req = SimpleNamespace(resolved_provider="auto", resolved_api_mode=None)
        _kind, _args, kwargs = aux._ladder_step_call(step, req, {"resolved_api_key": None}, {})
        client, _model = aux._get_cached_client(
            kwargs["resolved_provider"], kwargs["resolved_model"], api_key=kwargs["resolved_api_key"],
            main_runtime=MAIN_RUNTIME,
        )
        return client

    def test_retry_after_rotation_uses_the_rotated_entry_and_failures_stay_attributed(self, pool_home):
        import agent.auxiliary_client as aux
        pool_home([_entry("aaaa01", "A", 0, KEY_A), _entry("bbbb02", "B", 1, KEY_B),
                   _entry("cccc03", "C", 2, KEY_C)])
        aux._client_cache.clear()
        route, model = self._route(aux)

        ladder = aux._ladder_credential_rungs(_RateLimited(), route, {"model": model}, False)
        assert next(ladder).kind == "call"                     # same-key retry
        step = ladder.throw(_RateLimited())                    # still limited -> rotate
        assert step.kind == "retry_same_provider"
        assert self._retry_client(aux, step, model).api_key == KEY_B

        with pytest.raises(StopIteration):
            ladder.throw(_RateLimited())                       # B is limited too

        assert _status("A")[0] == "exhausted"
        assert _status("B")[0] == "exhausted"
        assert _status("C")[0] != "exhausted"
