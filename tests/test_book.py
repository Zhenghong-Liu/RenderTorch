import subprocess
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "book" / "build.py"
EXPECTED_SOURCES = {
    3: ["compute/camera.py", "compute/projection.py", "renderer/framebuffer.py"],
    4: ["renderer/clipping.py", "renderer/rasterizer.py", "app/viewport.py"],
    5: ["physics/common/state.py", "physics/common/simulation.py"],
    6: ["physics/cloth/mass_spring.py", "physics/cloth/geometry.py"],
    7: ["physics/cloth/pbd.py", "physics/cloth/geometry.py"],
    8: ["physics/cloth/xpbd.py", "physics/cloth/geometry.py"],
    10: ["physics/rigidbody/sphere.py", "physics/softbody/xpbd.py"],
}
REQUIRED_SECTIONS = (
    "intuition",
    "formula",
    "pseudocode",
    "source",
    "visual",
    "metrics",
    "debugging",
)


class BookParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.links = []
        self.scripts = []
        self.language = None
        self.title = ""
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if "id" in attributes:
            self.ids.add(attributes["id"])
        if tag == "a" and "href" in attributes:
            self.links.append(attributes["href"])
        if tag == "script" and "src" in attributes:
            self.scripts.append(attributes["src"])
        if tag == "html":
            self.language = attributes.get("lang")
        if tag == "title":
            self.in_title = True

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False

    def handle_data(self, data):
        if self.in_title:
            self.title += data


class BookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temporary.name) / "book"
        subprocess.run(
            [sys.executable, str(BUILD_SCRIPT), "--output", str(cls.output)],
            check=True,
            cwd=ROOT,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def parse(self, relative_path):
        parser = BookParser()
        parser.feed((self.output / relative_path).read_text(encoding="utf-8"))
        return parser

    def test_home_and_twelve_chapters_are_generated(self):
        self.assertTrue((self.output / "index.html").is_file())
        chapters = sorted((self.output / "chapters").glob("*.html"))
        self.assertEqual(len(chapters), 12)
        self.assertEqual(
            [path.name for path in chapters],
            [f"chapter{number:02d}.html" for number in range(1, 13)],
        )

    def test_every_page_is_chinese_and_has_the_required_teaching_sections(self):
        pages = [self.output / "index.html", *sorted((self.output / "chapters").glob("*.html"))]
        for page in pages:
            with self.subTest(page=page.name):
                parser = self.parse(page.relative_to(self.output))
                self.assertEqual(parser.language, "zh-CN")
                self.assertTrue(parser.title.strip())
                if page.name != "index.html":
                    for section in REQUIRED_SECTIONS:
                        self.assertIn(section, parser.ids)

    def test_cloth_chapters_link_to_real_algorithm_source(self):
        for number, paths in EXPECTED_SOURCES.items():
            with self.subTest(chapter=number):
                parser = self.parse(f"chapters/chapter{number:02d}.html")
                for path in paths:
                    self.assertTrue((ROOT / path).is_file(), path)
                    self.assertTrue(
                        any(path in link and "github.com/Zhenghong-Liu/RenderTorch" in link for link in parser.links),
                        path,
                    )

    def test_comparison_chapter_has_interactive_canvas_controls(self):
        content = (self.output / "chapters/chapter09.html").read_text(encoding="utf-8")
        self.assertIn('id="comparison-demo"', content)
        self.assertIn('id="play-comparison"', content)
        self.assertIn('id="reset-comparison"', content)
        self.assertIn('id="comparison-iterations"', content)
        self.assertEqual(content.count('data-algorithm="'), 3)
        self.assertIn('data-comparison-canvas', content)

    def test_local_assets_and_navigation_resolve(self):
        for page in self.output.rglob("*.html"):
            parser = self.parse(page.relative_to(self.output))
            for link in parser.links:
                if link.startswith(("https://", "http://", "#", "mailto:")):
                    continue
                target = (page.parent / link.split("#", 1)[0]).resolve()
                self.assertTrue(target.exists(), f"{page.name}: {link}")
            for script in parser.scripts:
                self.assertTrue((page.parent / script).is_file(), script)

    def test_styles_are_responsive_and_framework_free(self):
        css = (self.output / "assets/style.css").read_text(encoding="utf-8")
        self.assertIn("@media", css)
        self.assertIn("grid-template-columns", css)
        self.assertNotIn("tailwind", css.lower())
        self.assertNotIn("bootstrap", css.lower())
        javascript = (self.output / "assets/book.js").read_text(encoding="utf-8")
        self.assertIn("comparison-demo", javascript)
        self.assertIn("requestAnimationFrame", javascript)
        self.assertIn("clipboard", javascript)


if __name__ == "__main__":
    unittest.main()
