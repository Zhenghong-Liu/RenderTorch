import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
CONTENT_NAME = "rendertorch_physics_shared_content"
CONTENT_SPEC = importlib.util.spec_from_file_location(CONTENT_NAME, ROOT / "content.py")
CONTENT = importlib.util.module_from_spec(CONTENT_SPEC)
sys.modules[CONTENT_NAME] = CONTENT
CONTENT_SPEC.loader.exec_module(CONTENT)

Chapter = CONTENT.Chapter
DemoSpec = CONTENT.DemoSpec
Section = CONTENT.Section
SourceLink = CONTENT.SourceLink
CHAPTER_PATHS = tuple(ROOT / f"physics_content_chapter{number:02d}.py" for number in range(1, 11))
REFERENCES_PATH = ROOT / "physics_references.json"


def validate(chapter: Chapter) -> None:
    section_ids = [section.id for section in chapter.sections]
    demo_ids = [demo.id for demo in chapter.demos]
    if len(section_ids) != len(set(section_ids)):
        raise ValueError(f"{chapter.slug}: duplicate section id")
    if len(demo_ids) != len(set(demo_ids)):
        raise ValueError(f"{chapter.slug}: duplicate demo id")
    missing = [demo.id for demo in chapter.demos if demo.section_id not in section_ids]
    if missing:
        raise ValueError(f"{chapter.slug}: demo {missing[0]} targets unknown section {next(demo.section_id for demo in chapter.demos if demo.id == missing[0])}")
    seen = {"objectives", "sources"}
    for element_id in section_ids + demo_ids:
        if element_id in seen:
            raise ValueError(f"{chapter.slug}: duplicate HTML id: {element_id}")
        seen.add(element_id)


def _load_chapter(path: Path) -> Chapter:
    name = f"rendertorch_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    chapter = Chapter(
        *module.CHAPTER[:7],
        tuple(Section(*section) for section in module.CHAPTER[7]),
        tuple(DemoSpec(*demo) for demo in module.CHAPTER[8]),
        tuple(SourceLink(*source) for source in module.CHAPTER[9]),
    )
    validate(chapter)
    return chapter


def chapters():
    return tuple(_load_chapter(path) for path in CHAPTER_PATHS if path.is_file())


def references():
    if not REFERENCES_PATH.is_file():
        return []
    return json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
