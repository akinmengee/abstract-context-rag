from abstractrag.core.config import (
    IngestionSettings,
    PictureDescriptionSettings,
    VisionSettings,
    get_settings,
)


def test_picture_description_defaults_to_disabled():
    settings = IngestionSettings()
    assert settings.picture_description.enabled is False
    assert settings.picture_description.model == "granite3.2-vision:2b"


def test_picture_description_reads_config_yaml():
    settings = get_settings()
    assert isinstance(settings.ingestion.picture_description, PictureDescriptionSettings)
    assert settings.ingestion.picture_description.enabled is False


def test_figure_extraction_defaults_to_disabled():
    settings = IngestionSettings()
    assert settings.figures.enabled is False


def test_vision_defaults_to_disabled():
    settings = VisionSettings()
    assert settings.enabled is False
    assert settings.model == "granite3.2-vision:2b"


def test_vision_reads_config_yaml():
    settings = get_settings()
    assert isinstance(settings.vision, VisionSettings)
    assert settings.vision.enabled is False
