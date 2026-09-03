from app.config import get_settings

def test_get_settings_returns_the_same_cached_instance():
    settings_1 = get_settings()
    settings_2 = get_settings()

    assert settings_1 is settings_2