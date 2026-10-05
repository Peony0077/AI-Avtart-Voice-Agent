"""Owner and customer links, and who gets the one conversation (app/api/ws.py)."""
from types import SimpleNamespace

from app.api.ws import SessionManager, role_for


def test_tokens_decide_the_role():
    assert role_for("", None, None) == "owner"                         # local use without tokens
    assert role_for("own", "own", "cust") == "owner"
    assert role_for("cust", "own", "cust") == "customer"
    assert role_for("x", "own", "cust") is None and role_for("", "own", "cust") is None
    assert role_for("cust", "own", None) is None                        # no customer link configured


def test_nobody_cuts_off_a_customer_except_their_own_reconnect():
    s = lambda role, visitor: SimpleNamespace(role=role, visitor_id=visitor)   # noqa: E731
    takes = SessionManager.takes_over
    assert takes(s("customer", "a"), None)
    assert takes(s("owner", "o1"), s("owner", "o2"))                    # the owner's reload
    assert takes(s("customer", "a"), s("customer", "a"))                # the same browser reconnecting
    assert not takes(s("customer", "b"), s("customer", "a"))            # another customer waits
    assert not takes(s("owner", "o1"), s("customer", "a"))              # the owner waits too
    assert not takes(s("customer", None), s("customer", None))         # no id: never takes over
