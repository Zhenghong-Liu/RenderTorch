import importlib.util
import json
import os
import re
import subprocess
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
import html


BOOK_DIR = Path(os.environ.get("BOOK_DIR", "book"))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ContentContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.content = load_module("book_v2_content", BOOK_DIR / "content.py")

    def test_chapter_metadata_is_complete(self):
        rows = self.content.chapters()
        self.assertEqual([row.number for row in rows], list(range(1, 16)))
        self.assertEqual(len({row.slug for row in rows}), 15)
        for row in rows:
            self.assertTrue(row.title.strip())
            self.assertTrue(row.summary.strip())
            self.assertTrue(row.unit.strip())
            self.assertTrue(row.objectives)
            self.assertTrue(row.prerequisites)

    def test_reference_entries_have_license_and_usage(self):
        rows = self.content.references()
        self.assertGreaterEqual(len(rows), 8)
        for row in rows:
            for field in ("title", "authors", "url", "license", "usage"):
                self.assertTrue(row[field].strip(), (row.get("title"), field))


class BuildContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.content = load_module("book_v2_content", BOOK_DIR / "content.py")
        cls.build = load_module("book_v2_build", BOOK_DIR / "build.py")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.tmp.name)
        cls.build.build(cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_build_writes_home_references_and_fifteen_chapters(self):
        self.assertTrue((self.output / "index.html").is_file())
        self.assertTrue((self.output / "references.html").is_file())
        for row in self.content.chapters():
            self.assertTrue((self.output / "chapters" / f"{row.slug}.html").is_file())

    def test_latex_formulas_render_with_local_katex(self):
        formulas = []
        for page in sorted((self.output / "chapters").glob("chapter*.html")):
            text = page.read_text(encoding="utf-8")
            for block in re.findall(r'<div class="formula">(.*?)</div>', text, re.DOTALL):
                match = re.fullmatch(r"\s*\\\[(.*?)\\\]\s*", block, re.DOTALL)
                self.assertIsNotNone(match, f"{page.name} has a non-LaTeX formula block")
                formulas.append({"tex": html.unescape(match.group(1)), "display": True})
            formulas.extend(
                {"tex": html.unescape(tex), "display": False}
                for tex in re.findall(r"\\\((.*?)\\\)", text, re.DOTALL)
            )
        self.assertGreaterEqual(len(formulas), 20)

        renderer = BOOK_DIR / "assets" / "vendor" / "katex" / "katex.min.js"
        self.assertTrue(renderer.is_file())
        script = """
const fs = require('node:fs');
const katex = require(process.argv[1]);
const formulas = JSON.parse(fs.readFileSync(0, 'utf8'));
formulas.forEach((formula, index) => {
  try {
    const html = katex.renderToString(formula.tex, {
      displayMode: formula.display,
      throwOnError: true,
      strict: 'error',
    });
    if (!html.includes('class="katex"')) {
      throw new Error('formula did not render');
    }
  } catch (error) {
    console.error(`formula ${index}: ${error.message}`);
    process.exit(1);
  }
});
console.log(`Rendered ${formulas.length} LaTeX formulas`);
"""
        result = subprocess.run(
            ["node", "-e", script, str(renderer.resolve())],
            input=json.dumps(formulas),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"Rendered {len(formulas)} LaTeX formulas", result.stdout)

    def test_copy_buttons_use_shared_copy_handler(self):
        for page in sorted((self.output / "chapters").glob("chapter*.html")):
            buttons = re.findall(r'<button class="copy-button"[^>]*>', page.read_text(encoding="utf-8"))
            for button in buttons:
                self.assertIn("data-copy", button, (page.name, button))

    def test_pages_load_local_katex_assets_in_order(self):
        chapter = (self.output / "chapters" / "chapter01.html").read_text(encoding="utf-8")
        css = chapter.index('href="../assets/vendor/katex/katex.min.css"')
        renderer = chapter.index('src="../assets/vendor/katex/katex.min.js"')
        auto_render = chapter.index('src="../assets/vendor/katex/auto-render.min.js"')
        book = chapter.index('src="../assets/book.js"')
        self.assertLess(css, renderer)
        self.assertLess(renderer, auto_render)
        self.assertLess(auto_render, book)
        self.assertTrue((self.output / "assets/vendor/katex/katex.min.css").is_file())
        self.assertTrue((self.output / "assets/vendor/katex/LICENSE").is_file())
        css = (self.output / "assets/vendor/katex/katex.min.css").read_text(encoding="utf-8")
        font_paths = re.findall(r"url\((fonts/[^)]+)\)", css)
        self.assertTrue(font_paths)
        for font_path in font_paths:
            self.assertTrue((self.output / "assets/vendor/katex" / font_path).is_file(), font_path)

    def test_chapter_navigation_uses_parent_relative_paths(self):
        chapter = (self.output / "chapters" / "chapter01.html").read_text(encoding="utf-8")
        self.assertIn('href="../index.html"', chapter)
        self.assertIn('href="chapter02.html"', chapter)
        self.assertIn('href="chapter01.html" aria-current="page"', chapter)
        self.assertNotIn('href="../chapter', chapter)
        self.assertIn('href="../references.html"', chapter)
        self.assertNotIn('href="chapters/chapters/', chapter)
        self.assertNotIn('href="chapters/index.html"', chapter)

    def test_navigation_and_accessibility_markers_exist(self):
        home = (self.output / "index.html").read_text(encoding="utf-8")
        chapter = (self.output / "chapters" / "chapter01.html").read_text(encoding="utf-8")
        for marker in ('id="main-content"', "data-menu-toggle", 'class="skip-link"'):
            self.assertIn(marker, home)
            self.assertIn(marker, chapter)
        self.assertEqual(home.count('class="chapter-card"'), 15)
        self.assertIn("references.html", home)


class AssetParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.assets = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "a" and "href" in values:
            self.links.append(values["href"])
        if tag in {"script", "img"} and "src" in values:
            self.assets.append(values["src"])
        if tag == "link" and "href" in values:
            self.assets.append(values["href"])


class BuiltSiteTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.content = load_module("book_v2_content", BOOK_DIR / "content.py")
        cls.build = load_module("book_v2_build", BOOK_DIR / "build.py")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.tmp.name)
        cls.build.build(cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()


class LinkAndAssetTests(BuiltSiteTestCase):
    def test_every_local_link_and_asset_resolves(self):
        for page in self.output.rglob("*.html"):
            parser = AssetParser()
            parser.feed(page.read_text(encoding="utf-8"))
            for reference in parser.links + parser.assets:
                if reference.startswith(("https://", "http://", "mailto:", "#")):
                    continue
                path = reference.split("#", 1)[0].split("?", 1)[0]
                target = (page.parent / path).resolve()
                self.assertTrue(target.exists(), (page, reference))

    def test_pages_do_not_load_external_runtime_assets(self):
        for page in self.output.rglob("*.html"):
            parser = AssetParser()
            parser.feed(page.read_text(encoding="utf-8"))
            for reference in parser.assets:
                self.assertFalse(reference.startswith(("http://", "https://")), (page, reference))


class PageStructureTests(BuiltSiteTestCase):
    def test_chapter_pages_have_one_main_and_unique_demo_ids(self):
        seen = set()
        for row in self.content.chapters():
            page = self.output / "chapters" / f"{row.slug}.html"
            text = page.read_text(encoding="utf-8")
            self.assertEqual(text.count('id="main-content"'), 1)
            for demo in row.demos:
                self.assertNotIn(demo.id, seen)
                seen.add(demo.id)
                self.assertIn(f'id="{demo.id}"', text)
                self.assertIn(demo.script, text)

    def test_every_demo_has_reset_and_live_readout(self):
        for row in self.content.chapters():
            for demo in row.demos:
                script = (BOOK_DIR / "assets" / demo.script).read_text(encoding="utf-8")
                self.assertIn("data-reset", script, (row.slug, demo.script))
                self.assertIn('data-readout aria-live="polite"', script, (row.slug, demo.script))

    def test_reference_grid_can_shrink_and_wrap_long_urls(self):
        css = (BOOK_DIR / "assets" / "style.css").read_text(encoding="utf-8")
        mobile_rules = css[css.index("@media (max-width: 980px)"):css.index("@media (max-width: 720px)")]
        self.assertIn(".source-grid, .reference-grid { grid-template-columns: minmax(0, 1fr); }", mobile_rules)
        rule = re.search(r"\.reference-card \{([^}]+)\}", css)
        self.assertIsNotNone(rule)
        self.assertIn("min-width: 0", rule.group(1))
        self.assertIn("overflow-wrap: anywhere", rule.group(1))

    def test_print_reduced_motion_and_menu_focus_contracts(self):
        css = (BOOK_DIR / "assets" / "style.css").read_text(encoding="utf-8")
        javascript = (BOOK_DIR / "assets" / "book.js").read_text(encoding="utf-8")
        self.assertIn("@media print", css)
        self.assertIn("@media (prefers-reduced-motion: reduce)", css)
        self.assertIn(".demo-toolbar", css[css.index("@media print"):])
        self.assertIn(".page-nav", css[css.index("@media print"):])
        self.assertIn('sidebar.querySelector("a")?.focus()', javascript)
        self.assertIn('toggle.setAttribute("aria-expanded", "false")', javascript)


REQUIRED_SECTIONS = {
    "objectives", "intuition", "definitions", "derivation", "example",
    "pseudocode", "sources", "demo", "self-test", "debugging", "status",
}

UNIT_ONE = {
    1: ("pixel-demo", ("u &= \\frac{x}{W-1}", "RGBA")),
    2: ("vector-demo", ("\\mathbf a\\times\\mathbf b", "normalize")),
    3: ("transform-demo", ("\\mathbf p'=M\\mathbf p", "M^{-\\mathsf T}")),
    4: ("camera-demo", ("x_{\\mathrm{ndc}}", "x_{\\mathrm{pixel}}")),
}


class UnitContentTestCase(unittest.TestCase):
    unit_map = {}

    @classmethod
    def setUpClass(cls):
        cls.content = load_module("book_v2_content", BOOK_DIR / "content.py")
        cls.build = load_module("book_v2_build", BOOK_DIR / "build.py")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.tmp.name)
        cls.build.build(cls.output)
        cls.rows = {row.number: row for row in cls.content.chapters()}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_chapters_have_all_textbook_sections_and_formulas(self):
        for number in self.unit_map:
            row = self.rows[number]
            page = self.output / "chapters" / f"{row.slug}.html"
            text = page.read_text(encoding="utf-8")
            for section_id in REQUIRED_SECTIONS:
                self.assertIn(f'id="{section_id}"', text, row.title)
            demo_id, markers = self.unit_map[number]
            self.assertIn(f'id="{demo_id}"', text)
            self.assertTrue((BOOK_DIR / "assets" / row.demos[0].script).is_file())
            self.assertIn(row.demos[0].script, text)
            for marker in markers:
                self.assertIn(marker, text)

    def test_status_labels_do_not_overstate_implementation(self):
        for number in self.unit_map:
            row = self.rows[number]
            page = self.output / "chapters" / f"{row.slug}.html"
            text = page.read_text(encoding="utf-8")
            status_marker = text.index('id="status"')
            status_start = text.rfind("<section", 0, status_marker)
            status_end = text.index("</section>", status_marker)
            status = text[status_start:status_end]
            self.assertIn("浏览器概念实验", status)
            self.assertTrue(
                "现有 RenderTorch" in status or "后续实现路线" in status
            )


class UnitOneContentTests(UnitContentTestCase):
    unit_map = UNIT_ONE


UNIT_TWO = {
    5: ("mesh-demo", ("索引网格", "三角形扇")),
    6: ("topology-demo", ("V-E+F=2", "n_{\\mathrm{face}}")),
    7: ("curve-demo", ("de Casteljau", "B_i^n(t)")),
    8: ("surface-demo", ("S(u,v)", "\\frac{\\partial S}{\\partial u}")),
    9: ("scene-demo", ("\\mathbf p_{\\mathrm{world}}=M_{\\mathrm{model}}\\mathbf p_{\\mathrm{local}}", "AABB")),
}


class UnitTwoContentTests(UnitContentTestCase):
    unit_map = UNIT_TWO


UNIT_THREE = {
    10: ("raster-demo", ("E(a,b,p)", "\\lambda_0")),
    11: ("occlusion-demo", ("\\frac{1}{w_{\\mathrm{pixel}}}", "w_{\\mathrm{pixel}} < \\mathrm{depth}")),
}


class UnitThreeContentTests(UnitContentTestCase):
    unit_map = UNIT_THREE


UNIT_FOUR = {
    12: ("lighting-demo", ("diffuse = max(0, N · L)", "H = normalize(L + V)")),
    13: ("texture-demo", ("u_pixel", "w_pixel")),
    14: ("shadow-demo", ("C_linear^(1/2.2)", "C_reinhard")),
    15: ("raytrace-demo", ("p(t) = o + t d", "r = d − 2(d · n)n")),
}


class UnitFourContentTests(UnitContentTestCase):
    unit_map = UNIT_FOUR


if __name__ == "__main__":
    unittest.main()
