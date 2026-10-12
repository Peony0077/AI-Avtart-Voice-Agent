"""The profile can be chosen in .env (pydantic's .env source fills the settings, not os.environ)."""
import pytest

from app.core import config


def test_profile_from_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("AVATAR_PROFILE", raising=False)
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    (tmp_path / ".env").write_text("AVATAR_PROFILE=avatar\n", encoding="utf-8")
    assert config.profile_name() == "avatar"
    monkeypatch.setenv("AVATAR_PROFILE", "lite")            # the environment wins over .env
    assert config.profile_name() == "lite"


def test_the_profile_in_dotenv_is_the_one_loaded(monkeypatch):
    monkeypatch.setenv("AVATAR_PROFILE", "avatar")
    assert config.Settings(_env_file=None).avatar.enabled is True
    monkeypatch.setenv("AVATAR_PROFILE", "default")
    assert config.Settings(_env_file=None).avatar.enabled is False


def test_an_unknown_profile_is_an_error(monkeypatch):
    monkeypatch.setenv("AVATAR_PROFILE", "avatr")
    with pytest.raises(ValueError, match="unknown profile"):
        config.Settings(_env_file=None)
