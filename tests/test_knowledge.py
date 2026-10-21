from ecovision_api.config import settings
from ecovision_api.knowledge import KnowledgeBase


def test_hong_kong_glass_guidance_is_source_backed() -> None:
    guidance = KnowledgeBase(settings.knowledge_path).retrieve("hong_kong", "glass")
    assert any("remove caps" in statement.lower() for statement in guidance.statements)
    assert all(source["url"].startswith("https://") for source in guidance.sources)


def test_mainland_guidance_requests_city_specificity() -> None:
    guidance = KnowledgeBase(settings.knowledge_path).retrieve(
        "mainland_china", "plastic", city="Shenzhen"
    )
    assert guidance.statements[0] == "Location supplied by the user: Shenzhen."
