"""Owner and customer links, and who gets the one conversation (app/api/ws.py)."""
from types import SimpleNamespace

from app.api.ws import SessionManager, role_for


def test_tokens_decide_the_role():
    assert role_for("", None, None) == "owner"                         # local use without tokens
    assert role_for("own", "own", "cust") == "owner"
    assert role_for("cust", "own", "cust") == "customer"
    assert role_for("x", "own", "cust") is None and role_for("", "own", "cust") is None
    assert role_for("cust", "own", None) is None                        # no customer link configured
    # a customer link without an owner token: refused at start-up, and nobody is the owner here either
    assert role_for("", None, "cust") is None and role_for("cust", None, "cust") is None


def test_a_customer_link_without_an_owner_token_is_refused_at_startup():
    import pytest

    from app.core.config import Settings
    with pytest.raises(ValueError, match="access_token"):
        Settings(_env_file=None, customer_token="cust", access_token=None)
    assert Settings(_env_file=None, customer_token="cust", access_token="own").customer_token == "cust"


def test_nobody_cuts_off_a_customer_except_their_own_reconnect():
    s = lambda role, visitor: SimpleNamespace(role=role, visitor_id=visitor)   # noqa: E731
    takes = SessionManager.takes_over
    assert takes(s("customer", "a"), None)
    assert takes(s("owner", "o1"), s("owner", "o2"))                    # the owner's reload
    assert takes(s("customer", "a"), s("customer", "a"))                # the same browser reconnecting
    assert not takes(s("customer", "b"), s("customer", "a"))            # another customer waits
    assert not takes(s("owner", "o1"), s("customer", "a"))              # the owner waits too
    assert not takes(s("customer", None), s("customer", None))         # no id: never takes over
    # the owner's browser has a visitor id too: its customer-link tab is a customer, not the owner
    assert not takes(s("customer", "o1"), s("owner", "o1")) and not takes(s("owner", "o1"), s("customer", "o1"))
