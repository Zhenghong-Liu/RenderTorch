import html
import json
import re
import subprocess
import tempfile
import textwrap
import unittest
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

from book import build as graphics_build
from book.content import Chapter, DemoSpec, Section
from book import physics_build, physics_content


REPO_DIR = Path(__file__).resolve().parents[1]
CORE_PATH = REPO_DIR / "book" / "assets" / "physics-core.js"
UI_PATH = REPO_DIR / "book" / "assets" / "physics.js"


class HTMLTree(HTMLParser):
    VOID_ELEMENTS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self, text):
        super().__init__()
        self.elements = []
        self.stack = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        element = {"tag": tag, "attrs": dict(attrs), "parent": self.stack[-1] if self.stack else None}
        self.elements.append(element)
        if tag not in self.VOID_ELEMENTS:
            self.stack.append(element)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID_ELEMENTS:
            self.stack.pop()

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index]["tag"] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        for element in self.stack:
            element.setdefault("text", []).append(data)

    @staticmethod
    def text_content(element):
        return "".join(element.get("text", ()))


class PhysicsBuildContractTests(unittest.TestCase):
    @staticmethod
    def chapter(**changes):
        values = dict(
            number=1,
            slug="physics01",
            unit="单元一：运动会怎样发生",
            title="测试章",
            summary="用于验证构建器。",
            objectives=("识别状态",),
            prerequisites=(),
            sections=(
                Section("state-vectors", "速度向量", "可观察的运动", "<p>速度有方向和大小。</p>"),
                Section("units", "单位", "米每秒", "<p>速度单位是米每秒。</p>"),
            ),
            demos=(DemoSpec("particle-lab", "质点实验", "demos/physics01.js", "拖动质点。", "state-vectors"),),
            sources=(),
        )
        values.update(changes)
        return Chapter(**values)

    def test_display_math_scrolls_within_prose_on_narrow_pages(self):
        css = (REPO_DIR / "book" / "assets" / "style.css").read_text(encoding="utf-8")
        self.assertRegex(
            css,
            r"\.prose \.katex-display\s*\{[^}]*overflow-x:\s*auto",
            "Long display formulas must scroll locally instead of widening the document",
        )

    def test_inline_code_can_wrap_inside_prose(self):
        css = (REPO_DIR / "book" / "assets" / "style.css").read_text(encoding="utf-8")
        self.assertRegex(css, r"(?m)^code\s*\{[^}]*overflow-wrap:\s*anywhere")

    def test_source_links_can_shrink_and_wrap_long_paths(self):
        css = (REPO_DIR / "book" / "assets" / "style.css").read_text(encoding="utf-8")
        self.assertRegex(
            css,
            r"\.source-link\s*\{[^}]*min-width:\s*0[^}]*overflow-wrap:\s*anywhere",
        )

    def test_prose_tables_scroll_locally_on_narrow_pages(self):
        css = (REPO_DIR / "book" / "assets" / "style.css").read_text(encoding="utf-8")
        self.assertRegex(
            css,
            r"\.prose table\s*\{[^}]*display:\s*block[^}]*max-width:\s*100%[^}]*overflow-x:\s*auto",
        )

    def test_physics_pages_embed_demos_in_sections(self):
        from book import physics_build

        fixture = self.chapter()
        text = physics_build.render_chapter(fixture)
        self.assertIn('class="chapter-section-nav"', text)
        self.assertIn("1.1", text)
        self.assertIn("1.2", text)
        self.assertIn('id="particle-lab"', text)
        self.assertIn("../assets/physics-core.js", text)
        self.assertIn("../assets/physics.js", text)
        tree = HTMLTree(text)
        section = next(element for element in tree.elements if element["attrs"].get("id") == "state-vectors")
        demo = next(element for element in tree.elements if element["attrs"].get("id") == "particle-lab")
        self.assertIs(demo["parent"], section)
        section_children = [element for element in tree.elements if element["parent"] is section]
        self.assertEqual([child["attrs"].get("class") for child in section_children], [
            "section-label", "prose", "demo-mount physics-demo",
        ])
        scripts = [text.index(path) for path in (
            "../assets/vendor/katex/katex.min.js",
            "../assets/vendor/katex/auto-render.min.js",
            "../assets/book.js",
            "../assets/physics-core.js",
            "../assets/physics.js",
            "../assets/demos/physics01.js",
        )]
        self.assertEqual(scripts, sorted(scripts))

    def test_graphics_build_preserves_existing_physics_home(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            home = output / "physics" / "index.html"
            home.parent.mkdir(parents=True)
            home.write_text("formal physics home", encoding="utf-8")
            graphics_build.build(output)
            self.assertEqual(home.read_text(encoding="utf-8"), "formal physics home")

    def test_render_rejects_duplicate_section_and_demo_ids(self):
        from book import physics_build

        section = Section("same", "小节", "标题", "")
        with self.assertRaisesRegex(ValueError, "duplicate section"):
            physics_build.render_chapter(self.chapter(sections=(section, section)))
        demo = DemoSpec("same", "实验", "demos/test.js", "", "state-vectors")
        with self.assertRaisesRegex(ValueError, "duplicate demo"):
            physics_build.render_chapter(self.chapter(demos=(demo, demo)))

    def test_render_rejects_section_using_reserved_objectives_id(self):
        from book import physics_build

        section = Section("objectives", "小节", "标题", "")
        with self.assertRaisesRegex(ValueError, "duplicate HTML id: objectives"):
            physics_build.render_chapter(self.chapter(sections=(section,), demos=()))

    def test_render_rejects_section_using_reserved_sources_id(self):
        from book import physics_build

        section = Section("sources", "小节", "标题", "")
        with self.assertRaisesRegex(ValueError, "duplicate HTML id: sources"):
            physics_build.render_chapter(self.chapter(sections=(section,), demos=()))

    def test_render_rejects_section_and_demo_sharing_id(self):
        from book import physics_build

        section = Section("shared", "小节", "标题", "")
        demo = DemoSpec("shared", "实验", "demos/test.js", "", "shared")
        with self.assertRaisesRegex(ValueError, "duplicate HTML id: shared"):
            physics_build.render_chapter(self.chapter(sections=(section,), demos=(demo,)))

    def test_separate_chapters_may_reuse_section_ids(self):
        from book import physics_build

        section = Section("state", "小节", "标题", "")
        first = self.chapter(sections=(section,), demos=())
        second = self.chapter(number=2, slug="physics02", sections=(section,), demos=())
        physics_build.render_chapter(first)
        text = physics_build.render_chapter(second)
        self.assertIn('id="state"', text)

    def test_render_rejects_demo_targeting_missing_section(self):
        from book import physics_build

        demo = DemoSpec("missing-target", "实验", "demos/test.js", "", "absent")
        with self.assertRaisesRegex(ValueError, "unknown section"):
            physics_build.render_chapter(self.chapter(demos=(demo,)))

    def test_build_writes_subbook_pages_and_shared_assets(self):
        from book import physics_build

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(physics_build.CONTENT, "chapters", return_value=(self.chapter(),)):
                physics_build.build(output)
            home = (output / "physics" / "index.html").read_text(encoding="utf-8")
            chapter = (output / "physics" / "chapters" / "physics01.html").read_text(encoding="utf-8")
            references = output / "physics" / "references.html"
            self.assertIn('href="chapters/physics01.html"', home)
            self.assertIn('"../assets/style.css"', home)
            self.assertIn("../assets/physics.js", chapter)
            self.assertTrue(references.is_file())
            self.assertIn('href="chapters/physics01.html"', references.read_text(encoding="utf-8"))
            for asset in ("style.css", "physics-core.js", "physics.js"):
                self.assertTrue((output / "assets" / asset).is_file(), asset)
                self.assertTrue((output / "physics" / "assets" / asset).is_file(), asset)


class PhysicsSubbookContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.content = physics_content

    def test_physics_subbook_has_ten_chapters_and_two_demos_each(self):
        rows = self.content.chapters()
        self.assertEqual([row.number for row in rows], list(range(1, 11)))
        self.assertEqual(sum(len(row.demos) for row in rows), 20)
        for row in rows:
            self.assertGreaterEqual(len(row.sections), 10)
            self.assertGreaterEqual(len(row.demos), 2)
            for section in row.sections:
                self.assertGreaterEqual(len(section.html), 500, (row.slug, section.id))
                self.assertIn("<p>", section.html, (row.slug, section.id))
                self.assertIn("</p>", section.html, (row.slug, section.id))

    def test_all_physics_demo_scripts_exist(self):
        for row in physics_content.chapters():
            for demo in row.demos:
                script = REPO_DIR / "book" / "assets" / demo.script
                self.assertTrue(script.is_file(), demo.script)

    def test_all_physics_sources_exist(self):
        root = REPO_DIR
        for row in physics_content.chapters():
            for item in row.sources:
                self.assertTrue((root / item.path).is_file(), item.path)
                self.assertEqual(
                    item.url,
                    f"https://github.com/Zhenghong-Liu/RenderTorch/blob/feat/m1-qt-minimal-loop/{item.path}",
                )


class PhysicsReferenceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.content = physics_content

    def test_physics_references_cover_all_units(self):
        rows = self.content.references()
        self.assertGreaterEqual(len(rows), 6)
        for row in rows:
            for field in ("group", "title", "authors", "url", "license", "usage"):
                self.assertTrue(row[field].strip(), (row.get("title"), field))

    def test_physics_references_include_required_sources_and_groups(self):
        rows = self.content.references()
        self.assertEqual(
            {row["group"] for row in rows},
            {"基础力学", "约束与碰撞", "PBD/XPBD", "工程参考"},
        )
        required_urls = {
            "https://openstax.org/books/university-physics-volume-1/pages/preface",
            "https://www.cs.cmu.edu/~baraff/sigcourse/",
            "https://books.google.com/books?vid=ISBN9781568818770",
            "https://doi.org/10.1016/j.jvcir.2007.01.005",
            "https://doi.org/10.1145/1399504.1360662",
            "https://brm.io/matter-js/docs/files/src_core_Engine.js.html",
            "https://docs.blender.org/manual/en/5.3/physics/cloth/introduction.html",
        }
        self.assertTrue(required_urls <= {row["url"] for row in rows})
        for row in rows:
            self.assertIn("只核对知识，不复制正文/图片/代码", row["usage"])

    def test_reference_page_renders_ordered_groups_and_complete_cards(self):
        rows = self.content.references()
        text = physics_build.render_references()
        groups = ("基础力学", "约束与碰撞", "PBD/XPBD", "工程参考")
        positions = []
        expected_cards = []
        for number, group in enumerate(groups, 1):
            marker = f'<section class="section" id="reference-group-{number}">'
            self.assertIn(marker, text)
            start = text.index(marker)
            end = text.index("</section>", start)
            section = text[start:end]
            positions.append(start)
            self.assertIn(f"<h2>{group}</h2>", section)
            group_rows = [row for row in rows if row["group"] == group]
            expected_cards.extend(group_rows)
            self.assertEqual(section.count('class="reference-card"'), len(group_rows))
        self.assertEqual(positions, sorted(positions))

        tree = HTMLTree(text)
        cards = [
            element for element in tree.elements
            if "reference-card" in element["attrs"].get("class", "").split()
        ]
        self.assertEqual(len(cards), len(expected_cards))

        def is_descendant(element, card):
            parent = element["parent"]
            while parent is not None:
                if parent is card:
                    return True
                parent = parent["parent"]
            return False

        for row, card in zip(expected_cards, cards):
            with self.subTest(title=row["title"]):
                children = [element for element in tree.elements if element["parent"] is card]
                headings = [element for element in children if element["tag"] == "h2"]
                self.assertEqual([HTMLTree.text_content(element) for element in headings], [row["title"]])
                authors = [
                    element for element in children
                    if element["tag"] == "p"
                    and "reference-authors" in element["attrs"].get("class", "").split()
                ]
                self.assertEqual([HTMLTree.text_content(element) for element in authors], [row["authors"]])
                links = [
                    element for element in tree.elements
                    if element["tag"] == "a" and is_descendant(element, card)
                ]
                self.assertEqual(
                    [(element["attrs"].get("href"), HTMLTree.text_content(element)) for element in links],
                    [(row["url"], row["url"])],
                )
                statuses = [
                    element for element in tree.elements
                    if "status-pill" in element["attrs"].get("class", "").split()
                    and is_descendant(element, card)
                ]
                self.assertEqual(
                    [HTMLTree.text_content(element) for element in statuses],
                    [f'许可：{row["license"]}'],
                )
                paragraphs = [element for element in children if element["tag"] == "p"]
                self.assertIn(row["usage"], [HTMLTree.text_content(element) for element in paragraphs])
        self.assertNotIn("将在全书内容完成后加入", text)

    def test_reference_page_retains_local_katex_mit_notice(self):
        text = physics_build.render_references()
        self.assertIn("KaTeX 0.16.47", text)
        self.assertIn("MIT", text)
        self.assertIn('href="../assets/vendor/katex/LICENSE"', text)
        self.assertTrue((REPO_DIR / "book/assets/vendor/katex/LICENSE").is_file())

    def test_reference_page_rejects_malformed_records(self):
        valid = {
            "group": "基础力学",
            "title": "测试资料",
            "authors": "测试作者",
            "url": "https://example.com/reference",
            "license": "测试许可",
            "usage": "只核对知识，不复制正文/图片/代码。",
        }
        cases = [
            ("root-not-list", {}, "must be a list"),
            ("record-not-object", [None], "must be an object"),
        ]
        for field in ("group", "title", "authors", "url", "license", "usage"):
            missing = dict(valid)
            del missing[field]
            cases.extend([
                (f"{field}-missing", [missing], f"invalid or missing {field}"),
                (f"{field}-blank", [dict(valid, **{field: " \t\n "})], f"invalid or missing {field}"),
                (f"{field}-non-string", [dict(valid, **{field: None})], f"invalid or missing {field}"),
            ])
        for label, rows, message in cases:
            with self.subTest(case=label):
                with patch.object(physics_build.CONTENT, "references", return_value=rows):
                    with self.assertRaisesRegex(ValueError, message):
                        physics_build.render_references()

    def test_reference_page_rejects_unknown_or_missing_groups(self):
        valid = {
            "group": "基础力学",
            "title": "测试资料",
            "authors": "测试作者",
            "url": "https://example.com/reference",
            "license": "测试许可",
            "usage": "只核对知识，不复制正文/图片/代码。",
        }
        with patch.object(physics_build.CONTENT, "references", return_value=[dict(valid, group="未知分组")]):
            with self.assertRaisesRegex(ValueError, "unknown reference group"):
                physics_build.render_references()
        with patch.object(physics_build.CONTENT, "references", return_value=[valid]):
            with self.assertRaisesRegex(ValueError, "missing reference group"):
                physics_build.render_references()


class PhysicsChapterContentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tempdir.cleanup)
        cls.output = Path(cls.tempdir.name)
        physics_build.build(cls.output)
        cls.rows = {row.number: row for row in physics_content.chapters()}

    def test_chapter_one_teaches_state_vectors_and_units(self):
        row = self.rows[1]
        text = (self.output / "physics" / "chapters" / "physics01.html").read_text(encoding="utf-8")
        for marker in ("质点", "单位", "速度", "\\mathbf v", "状态"):
            self.assertIn(marker, text)
        self.assertEqual({demo.id for demo in row.demos}, {"particle-lab", "frame-states"})

    def test_chapter_one_has_complete_numbered_teaching_and_demo_targets(self):
        row = self.rows[1]
        self.assertEqual((row.slug, row.unit, row.title),
                         ("physics01", "单元一：运动会怎样发生", "从真实运动到数字状态"))
        self.assertGreaterEqual(len(row.sections), 10)
        self.assertEqual([demo.section_id for demo in row.demos],
                         ["velocity-vector", "discrete-frames"])
        ids = [section.id for section in row.sections] + [demo.id for demo in row.demos]
        self.assertEqual(len(ids), len(set(ids)))
        text = (self.output / "physics" / "chapters" / "physics01.html").read_text(encoding="utf-8")
        tree = HTMLTree(text)
        for demo_id in ("particle-lab", "frame-states"):
            demo = next(element for element in tree.elements if element["attrs"].get("id") == demo_id)
            section = next(element for element in tree.elements
                           if element["attrs"].get("id") == demo["parent"]["attrs"]["id"])
            children = [child for child in tree.elements if child["parent"] is section]
            self.assertEqual([child["attrs"].get("class") for child in children],
                             ["section-label", "prose", "demo-mount physics-demo"])
        for marker in ("p_0=(1,2)", "v=(2,-1)", "dt=0.5", "p=(2,1.5)",
                       "伪代码", "自测", "调试", "浏览器概念实验", "当前 Python 实现"):
            self.assertIn(marker, text)

    def test_chapter_two_teaches_velocity_acceleration_and_integration(self):
        self.assertIn(2, self.rows, "chapter 2 must be available")
        row = self.rows[2]
        text = (self.output / "physics" / "chapters" / "physics02.html").read_text(encoding="utf-8")
        self.assertEqual(row.slug, "physics02")
        for marker in ("前向欧拉", "半隐式欧拉", "固定时间步", r"a=\frac{\Delta v}{\Delta t}"):
            self.assertIn(marker, text)
        self.assertEqual({demo.id for demo in row.demos}, {"integration-race", "fixed-step"})

    def test_chapter_two_has_eleven_sections_example_and_demo_targets(self):
        self.assertIn(2, self.rows, "chapter 2 must be available")
        row = self.rows[2]
        self.assertEqual((row.unit, row.title),
                         ("单元一：运动会怎样发生", "速度、加速度与时间积分"))
        self.assertEqual(len(row.sections), 11)
        self.assertEqual([demo.section_id for demo in row.demos],
                         ["exact-projectile", "fixed-timestep"])
        ids = [section.id for section in row.sections] + [demo.id for demo in row.demos]
        self.assertEqual(len(ids), len(set(ids)))
        text = (self.output / "physics" / "chapters" / "physics02.html").read_text(encoding="utf-8")
        tree = HTMLTree(text)
        for demo_id, section_id in (("integration-race", "exact-projectile"),
                                    ("fixed-step", "fixed-timestep")):
            demo = next(element for element in tree.elements if element["attrs"].get("id") == demo_id)
            section = next(element for element in tree.elements if element["attrs"].get("id") == section_id)
            self.assertIs(demo["parent"], section)
            children = [child for child in tree.elements if child["parent"] is section]
            self.assertEqual([child["attrs"].get("class") for child in children],
                             ["section-label", "prose", "demo-mount physics-demo"])
        for marker in ("fixed_dt=1/120", "substeps=2", "h=1/240", "速度重建",
                       "g=-9.81", "v0=20", "dt=0.1", "19.019", "1.9019", "1.95095",
                       "累加器", "伪代码", "调试", "浏览器概念实验", "当前 Python 实现",
                       "暂停", "单步"):
            self.assertIn(marker, text)

    def test_chapter_three_teaches_forces_energy_and_damping(self):
        self.assertIn(3, self.rows, "chapter 3 must be available")
        row = self.rows[3]
        text = (self.output / "physics" / "chapters" / "physics03.html").read_text(encoding="utf-8")
        self.assertEqual((row.slug, row.unit, row.title),
                         ("physics03", "单元一：运动会怎样发生", "力、能量与阻尼"))
        self.assertEqual(len(row.sections), 10)
        for marker in ("重力", "空气阻力", "弹簧力", "动能", "势能", "终端速度",
                       r"E=\frac12mv^2", "功", "阻尼", "伪代码", "能量诊断"):
            self.assertIn(marker, text)
        self.assertEqual([demo.id for demo in row.demos], ["force-balance", "energy-ledger"])
        self.assertEqual([demo.section_id for demo in row.demos], ["air-drag", "total-energy"])
        self.assertEqual([demo.script for demo in row.demos],
                         ["demos/physics05.js", "demos/physics06.js"])
        ids = [section.id for section in row.sections] + [demo.id for demo in row.demos]
        self.assertEqual(len(ids), len(set(ids)))

    def test_chapter_three_documents_models_numerics_and_real_sources(self):
        row = self.rows[3]
        text = (self.output / "physics" / "chapters" / "physics03.html").read_text(encoding="utf-8")
        tree = HTMLTree(text)
        for demo_id, section_id in (("force-balance", "air-drag"), ("energy-ledger", "total-energy")):
            demo = next(element for element in tree.elements if element["attrs"].get("id") == demo_id)
            section = next(element for element in tree.elements if element["attrs"].get("id") == section_id)
            self.assertIs(demo["parent"], section)
            self.assertEqual([child["attrs"].get("class") for child in tree.elements
                              if child["parent"] is section],
                             ["section-label", "prose", "demo-mount physics-demo"])
        for marker in ("线性阻力是模型近似", "双精度", "float32", r"v_t=\frac{mg}{c}",
                       r"v_t=\frac{mg-F_b}{c}", r"\frac{E_{\mathrm{initial}}-E_{\mathrm{current}}}{E_{\mathrm{initial}}}",
                       "m=1", "g=9.81", "c=2", "4.905", "稳定步长", "不保证",
                       "浏览器概念实验", "当前 Python 实现"):
            self.assertIn(marker, text)
        source_paths = {source.path for source in row.sources}
        self.assertTrue({"physics/common/simulation.py", "physics/cloth/mass_spring.py",
                         "book/assets/physics-core.js"} <= source_paths)
        for source in row.sources:
            self.assertTrue((REPO_DIR / source.path).is_file(), source.path)
            self.assertNotIn("book-v2", source.path)

    def test_chapter_four_teaches_eleven_mass_spring_sections(self):
        self.assertIn(4, self.rows, "chapter 4 must be available")
        row = self.rows[4]
        text = (self.output / "physics" / "chapters" / "physics04.html").read_text(encoding="utf-8")
        self.assertEqual((row.slug, row.unit, row.title),
                         ("physics04", "单元二：物体为什么保持形状", "质量弹簧系统"))
        self.assertEqual(len(row.sections), 11)
        for marker in ("静止长度", "胡克定律", "F=-kx", "单位方向", "等大反向力",
                       "相对速度", "单振子", "稳定步长", "dt &lt; 2/sqrt(k/m)",
                       "网格", "结构", "剪切", "弯曲", "固定点", "碰撞",
                       "k=20", "m=1", "x0=0.5", "伪代码", "调试"):
            self.assertIn(marker, text)
        self.assertEqual([demo.id for demo in row.demos], ["spring-oscillator", "spring-cloth"])
        self.assertEqual([demo.section_id for demo in row.demos], ["oscillator", "spring-mesh"])
        self.assertEqual([demo.script for demo in row.demos],
                         ["demos/physics07.js", "demos/physics08.js"])
        ids = [section.id for section in row.sections] + [demo.id for demo in row.demos]
        self.assertEqual(len(ids), len(set(ids)))

    def test_chapter_four_separates_equilibrium_period_models_and_demo_scope(self):
        row = self.rows[4]
        text = (self.output / "physics" / "chapters" / "physics04.html").read_text(encoding="utf-8")
        tree = HTMLTree(text)
        for demo_id, section_id in (("spring-oscillator", "oscillator"),
                                    ("spring-cloth", "spring-mesh")):
            demo = next(element for element in tree.elements if element["attrs"].get("id") == demo_id)
            section = next(element for element in tree.elements
                           if element["attrs"].get("id") == section_id)
            self.assertIs(demo["parent"], section)
            self.assertEqual([child["attrs"].get("class") for child in tree.elements
                              if child["parent"] is section],
                             ["section-label", "prose", "demo-mount physics-demo"])
        for marker in ("静平衡", "mg/k", "不再重复加重力", "小振幅无阻尼周期近似",
                       "剪切弹簧", "弯曲弹簧", "本 Demo 只启用结构弹簧",
                       "自碰撞", "双精度", "float32", "浏览器概念实验", "当前 Python 实现"):
            self.assertIn(marker, text)
        source_paths = {source.path for source in row.sources}
        self.assertTrue({"physics/cloth/geometry.py", "physics/cloth/mass_spring.py",
                         "physics/common/simulation.py", "book/assets/physics-core.js"} <= source_paths)
        for source in row.sources:
            self.assertTrue((REPO_DIR / source.path).is_file(), source.path)
            self.assertNotIn("book-v2", source.path)

    def test_chapter_five_teaches_eleven_pbd_sections_and_exact_worked_projection(self):
        self.assertIn(5, self.rows, "chapter 5 must be available")
        row = self.rows[5]
        text = (self.output / "physics" / "chapters" / "physics05.html").read_text(encoding="utf-8")
        self.assertEqual((row.slug, row.unit, row.title),
                         ("physics05", "单元二：物体为什么保持形状", "PBD：直接修正位置"))
        self.assertEqual(len(row.sections), 11)
        for marker in ("力", "位置修正", "距离约束", "梯度", "单次投影", "逆质量", "固定点",
                       "迭代", "Gauss–Seidel", "速度重建", "伪代码", "调试",
                       r"C=\lVert\mathbf p_b-\mathbf p_a\rVert-L_0", r"w_i=1/m_i",
                       "a=(0,0)", "b=(3,0)", "L_0=2", "w_a=w_b=1",
                       "a=(0.5,0)", "b=(2.5,0)"):
            self.assertIn(marker, text)
        self.assertEqual([demo.id for demo in row.demos], ["pbd-rope", "constraint-order"])
        self.assertEqual([demo.section_id for demo in row.demos], ["projection", "gauss-seidel"])
        self.assertEqual([demo.script for demo in row.demos],
                         ["demos/physics09.js", "demos/physics10.js"])
        ids = [section.id for section in row.sections] + [demo.id for demo in row.demos]
        self.assertEqual(len(ids), len(set(ids)))

    def test_chapter_five_uses_independent_order_clones_and_position_correction_language(self):
        row = self.rows[5]
        sections = {section.id: section for section in row.sections}
        pseudocode = sections["pseudocode-debugging"].html
        self.assertNotIn("forward = reverse = alternating = sameInitialState()", pseudocode)
        for line in (
            "initial = sameInitialState()",
            "forward = clone(initial)",
            "reverse = clone(initial)",
            "alternating = clone(initial)",
        ):
            self.assertIn(line, pseudocode)
        self.assertEqual(pseudocode.count("clone(initial)"), 3)

        inverse_mass = sections["inverse-mass"].html
        self.assertIn("同样大小的位置修正", inverse_mass)
        self.assertIn("位置位移等量反向", inverse_mass)
        self.assertNotIn("冲量", inverse_mass)
        self.assertIn(r"w_a\Delta\mathbf p_{\mathrm{corr}}", inverse_mass)
        self.assertIn(r"只有当 \(w_a=w_b\) 时两端移动距离才相等", inverse_mass)

        worked = sections["worked-example"]
        self.assertEqual(worked.title, "算例：把两点投影到指定距离")
        self.assertNotIn("(0,0)", worked.title)
        self.assertIn(r"a'=a+w_a\Delta\mathbf p_{\mathrm{corr}}=(0.5,0)", worked.html)
        self.assertIn(r"b'=b-w_b\Delta\mathbf p_{\mathrm{corr}}=(2.5,0)", worked.html)

    def test_chapter_five_limits_precision_scope_and_maps_real_sources(self):
        self.assertIn(5, self.rows, "chapter 5 must be available")
        row = self.rows[5]
        text = (self.output / "physics" / "chapters" / "physics05.html").read_text(encoding="utf-8")
        for marker in ("最大相对约束误差", "固定点", "刚度小于 1", "双精度", "float32",
                       "结构（structural）", "剪切（shear）", "弯曲（bend）",
                       "fixed_dt=1/120", "substeps=2", "浏览器概念实验", "不是生产引擎"):
            self.assertIn(marker, text)
        source_paths = {source.path for source in row.sources}
        self.assertEqual(source_paths, {"physics/cloth/pbd.py", "physics/common/simulation.py",
                                       "physics/common/state.py", "book/assets/physics-core.js"})
        for source in row.sources:
            self.assertTrue((REPO_DIR / source.path).is_file(), source.path)
            self.assertIn("/blob/feat/m1-qt-minimal-loop/", source.url)
        tree = HTMLTree(text)
        for demo_id, section_id in (("pbd-rope", "projection"), ("constraint-order", "gauss-seidel")):
            demo = next(element for element in tree.elements if element["attrs"].get("id") == demo_id)
            section = next(element for element in tree.elements
                           if element["attrs"].get("id") == demo["parent"]["attrs"]["id"])
            self.assertEqual(section["attrs"]["id"], section_id)
            self.assertEqual([child["attrs"].get("class") for child in tree.elements
                              if child["parent"] is section],
                             ["section-label", "prose", "demo-mount physics-demo"])


    def test_chapter_six_teaches_collision_impulse_friction_and_continuous_detection(self):
        self.assertIn(6, self.rows, "chapter 6 must be available")
        row = self.rows[6]
        text = (self.output / "physics" / "chapters" / "physics06.html").read_text(encoding="utf-8")
        self.assertEqual((row.slug, row.unit, row.title),
                         ("physics06", "单元二：物体为什么保持形状", "碰撞、恢复与摩擦"))
        self.assertEqual(len(row.sections), 11)
        for marker in ("碰撞法线", "冲量", "恢复系数", "摩擦", "穿透", "位置投影",
                       "相对速度", "离散", "swept AABB", "进入时间", "离开时间",
                       "vy=-5", "e=0.5", "vy'=2.5", "伪代码", "调试",
                       "max(0,1-friction)", "切向冲量上限", "公共粒子碰撞路径"):
            self.assertIn(marker, text)
        self.assertEqual([demo.id for demo in row.demos], ["collision-friction", "continuous-collision"])
        self.assertEqual([demo.section_id for demo in row.demos],
                         ["tangential-friction", "continuous-detection"])
        self.assertEqual([demo.script for demo in row.demos],
                         ["demos/physics11.js", "demos/physics12.js"])
        ids = [section.id for section in row.sections] + [demo.id for demo in row.demos]
        self.assertEqual(len(ids), len(set(ids)))
        sections = {section.id: section.html for section in row.sections}
        self.assertIn("陈旧", sections["tangential-friction"])
        self.assertIn("approaching-only", sections["normal-impulse"])
        self.assertIn("不充当正确冲量范例", sections["pseudocode-debugging"])
        self.assertIn(r"t_{\mathrm{enter}}", sections["continuous-detection"])
        self.assertIn(r"t_{\mathrm{exit}}", sections["continuous-detection"])
        source_paths = {source.path for source in row.sources}
        self.assertEqual(source_paths, {"physics/common/colliders.py", "physics/rigidbody/sphere.py",
                                       "physics/common/simulation.py", "book/assets/physics-core.js"})
        for source in row.sources:
            self.assertTrue((REPO_DIR / source.path).is_file(), source.path)
            self.assertIn("/blob/feat/m1-qt-minimal-loop/", source.url)
        tree = HTMLTree(text)
        for demo_id, section_id in (("collision-friction", "tangential-friction"),
                                    ("continuous-collision", "continuous-detection")):
            demo = next(element for element in tree.elements if element["attrs"].get("id") == demo_id)
            section = next(element for element in tree.elements if element["attrs"].get("id") == section_id)
            self.assertIs(demo["parent"], section)
            self.assertEqual([child["attrs"].get("class") for child in tree.elements
                              if child["parent"] is section],
                             ["section-label", "prose", "demo-mount physics-demo"])

    def test_chapter_six_maps_real_python_static_sphere_friction(self):
        import torch
        from physics.common.colliders import SphereCollider
        from physics.rigidbody.sphere import RigidBodySimulation, SphereBody

        collider = SphereCollider(center=(0, 0, 0), radius=0.3,
                                  restitution=0.5, friction=0.2)
        positions = torch.tensor([[0.0, 0.2, 0.0]], dtype=torch.float32)
        velocities = torch.tensor([[4.0, -5.0, 0.0]], dtype=torch.float32)
        inverse_mass = torch.tensor([1.0], dtype=torch.float32)
        self.assertEqual(collider.project(positions, velocities, inverse_mass), 1)
        for actual, expected in zip(positions[0].tolist(), (0.0, 0.3, 0.0)):
            self.assertAlmostEqual(actual, expected, places=6)
        for actual, expected in zip(velocities[0].tolist(), (3.2, 1.0, 0.0)):
            self.assertAlmostEqual(actual, expected, places=6)

        body = SphereBody(position=torch.tensor([0.0, 0.2, 0.0], dtype=torch.float32),
                          velocity=torch.tensor([4.0, -5.0, 0.0], dtype=torch.float32),
                          radius=0.5, inverse_mass=1.0)
        simulation = RigidBodySimulation(body, gravity=(0, 0, 0))
        simulation._collide_static_sphere(collider)
        for actual, expected in zip(body.position.tolist(), (0.0, 0.8, 0.0)):
            self.assertAlmostEqual(actual, expected, places=6)
        for actual, expected in zip(body.velocity.tolist(), (3.2, 2.5, 0.0)):
            self.assertAlmostEqual(actual, expected, places=6)

        friction = {section.id: section.html for section in self.rows[6].sections}["tangential-friction"]
        self.assertIn("只在法向速度小于零时施加法向响应，并在响应后重新读取速度计算切向", friction)
        self.assertIn("陈旧切向分解只属于公共 <code>SphereCollider.project</code>", friction)
        self.assertIn("即使已经分离，只要位置仍穿透，摩擦分支仍会执行", friction)
        self.assertIn("没有像浏览器 <code>max(0,1-friction)</code> 那样钳制到非负", friction)
        self.assertNotIn("<code>_collide_static_sphere</code> 也保存碰撞前", friction)

    def test_chapter_seven_teaches_eleven_xpbd_sections_and_exact_demo_targets(self):
        self.assertIn(7, self.rows, "chapter 7 must be available")
        row = self.rows[7]
        text = (self.output / "physics" / "chapters" / "physics07.html").read_text(encoding="utf-8")
        self.assertEqual((row.slug, row.unit, row.title),
                         ("physics07", "单元三：让求解器尊重材料与时间", "XPBD：用柔度表达软硬"))
        self.assertEqual(len(row.sections), 11)
        self.assertEqual([demo.id for demo in row.demos], ["xpbd-compliance", "area-constraint"])
        self.assertEqual([demo.section_id for demo in row.demos], ["compliance", "volume-gradient"])
        self.assertEqual([demo.script for demo in row.demos],
                         ["demos/physics13.js", "demos/physics14.js"])
        ids = [section.id for section in row.sections] + [demo.id for demo in row.demos]
        self.assertEqual(len(ids), len(set(ids)))
        for marker in ("PBD", "刚度", "约束势能", "柔度", "有效柔度", "乘子", "增量更新",
                       "生命周期", "距离 XPBD", "面积", "体积", "梯度", "子步不变性",
                       r"\tilde{\alpha}=\alpha/h^2", r"\Delta\lambda", "alpha=0.01",
                       "h=1/60", "36", "伪代码", "调试", "浏览器概念实验", "当前 Python 实现"):
            self.assertIn(marker, text)
        tree = HTMLTree(text)
        for demo_id, section_id in (("xpbd-compliance", "compliance"),
                                    ("area-constraint", "volume-gradient")):
            demo = next(element for element in tree.elements if element["attrs"].get("id") == demo_id)
            section = next(element for element in tree.elements
                           if element["attrs"].get("id") == demo["parent"]["attrs"]["id"])
            self.assertEqual(section["attrs"]["id"], section_id)
            self.assertEqual([child["attrs"].get("class") for child in tree.elements
                              if child["parent"] is section],
                             ["section-label", "prose", "demo-mount physics-demo"])

    def test_chapter_seven_separates_lambda_geometry_substeps_and_implementation_boundaries(self):
        self.assertIn(7, self.rows, "chapter 7 must be available")
        row = self.rows[7]
        sections = {section.id: section.html for section in row.sections}
        self.assertIn(r"\tilde{\alpha}=\frac{0.01}{(1/60)^2}", sections["effective-compliance"])
        self.assertIn("=36", sections["effective-compliance"])
        self.assertIn(r"\Delta\lambda", sections["incremental-update"])
        self.assertIn(r"\frac{-C-\tilde{\alpha}\lambda}", sections["incremental-update"])
        self.assertIn(r"W+\tilde{\alpha}", sections["incremental-update"])
        self.assertIn("每个实际子步开始", sections["lambda-lifecycle"])
        self.assertIn("同一子步的所有迭代", sections["lambda-lifecycle"])
        self.assertIn("近似", sections["substep-invariance"])
        self.assertIn("不保证逐位相同", sections["substep-invariance"])
        self.assertIn("二维四粒子面积类比", sections["volume-gradient"])
        self.assertIn("原始环绕方向", sections["volume-gradient"])
        self.assertIn("projectAreaXPBD", sections["volume-gradient"])
        self.assertIn("没有面积约束", sections["pseudocode-debugging"])
        self.assertIn("没有独立弯曲约束", sections["pseudocode-debugging"])
        self.assertIn("没有自碰撞", sections["pseudocode-debugging"])
        self.assertIn("没有中心粒子参与体积", sections["pseudocode-debugging"])
        self.assertIn("六个四面体", sections["pseudocode-debugging"])
        self.assertIn("不能据此声称已有精细体积网格", sections["pseudocode-debugging"])
        table = sections["pseudocode-debugging"]
        self.assertIn("<table>", table)
        self.assertIn("症状", table)
        self.assertIn("优先检查", table)
        source_paths = {source.path for source in row.sources}
        self.assertEqual(source_paths, {
            "physics/cloth/pbd.py", "physics/cloth/xpbd.py", "physics/common/simulation.py",
            "physics/softbody/geometry.py", "physics/softbody/xpbd.py", "book/assets/physics-core.js",
        })
        for source in row.sources:
            self.assertTrue((REPO_DIR / source.path).is_file(), source.path)
            self.assertIn("/blob/feat/m1-qt-minimal-loop/", source.url)


class PhysicsChapterEightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tempdir.cleanup)
        cls.output = Path(cls.tempdir.name)
        physics_build.build(cls.output)
        cls.rows = {row.number: row for row in physics_content.chapters()}

    def test_chapter_eight_has_twelve_substantial_sections_and_exact_demo_bindings(self):
        self.assertIn(8, self.rows, "chapter 8 must be available")
        row = self.rows[8]
        self.assertEqual((row.slug, row.unit, row.title),
                         ("physics08", "单元三：让求解器尊重材料与时间", "三种算法同题比较"))
        self.assertEqual(len(row.sections), 12)
        self.assertEqual([section.id for section in row.sections], [
            "controlled-setup", "shared-loop", "stretch-response", "parameter-mapping",
            "iteration-substeps", "energy-damping", "constraint-error", "timestep-change",
            "performance-metrics", "worked-example", "decision-table", "pseudocode-debugging",
        ])
        for section in row.sections:
            prose = HTMLTree(section.html)
            self.assertGreater(len(section.html), 500, section.id)
            self.assertIn("<p>", section.html, section.id)
            self.assertIn("</p>", section.html, section.id)
        self.assertEqual([demo.id for demo in row.demos], ["solver-race", "substep-invariance"])
        self.assertEqual([demo.section_id for demo in row.demos],
                         ["stretch-response", "timestep-change"])
        self.assertEqual([demo.script for demo in row.demos],
                         ["demos/physics15.js", "demos/physics16.js"])
        ids = [section.id for section in row.sections] + [demo.id for demo in row.demos]
        self.assertEqual(len(ids), len(set(ids)))
        text = (self.output / "physics/chapters/physics08.html").read_text(encoding="utf-8")
        for number in range(1, 13):
            self.assertIn(f"8.{number}", text)
        tree = HTMLTree(text)
        for demo_id, section_id in (("solver-race", "stretch-response"),
                                    ("substep-invariance", "timestep-change")):
            demo = next(element for element in tree.elements if element["attrs"].get("id") == demo_id)
            section = next(element for element in tree.elements
                           if element["attrs"].get("id") == demo["parent"]["attrs"]["id"])
            self.assertEqual(section["attrs"]["id"], section_id)
            self.assertEqual([child["attrs"].get("class") for child in tree.elements
                              if child["parent"] is section],
                             ["section-label", "prose", "demo-mount physics-demo"])

    def test_chapter_eight_teaches_controlled_comparison_limits_and_real_sources(self):
        row = self.rows[8]
        sections = {section.id: section.html for section in row.sections}
        for marker in ("公平比较", "质量弹簧", "PBD", "XPBD", "子步不变性", "约束误差"):
            self.assertIn(marker, text := "".join(section.html for section in row.sections))
        for marker in ("共同初始条件", "共同循环", "速度重建", "动能", "弹性势能",
                       "有限迭代", "非线性", "阻尼", "参数不可直接互换", "性能指标",
                       "决策表", "伪代码", "调试", "不保证逐位相同", "没有通用赢家"):
            self.assertIn(marker, sections["pseudocode-debugging"] + "".join(sections.values()))
        self.assertIn(r"\(K=\frac12m\lVert\mathbf v\rVert^2\)", sections["energy-damping"])
        self.assertIn(r"U=\frac12k(L-L_0)^2", sections["energy-damping"])
        self.assertIn(r"E=\max_t\frac{|L(t)-L_0|}{L_0}", sections["constraint-error"])
        self.assertIn(r"\frac{1}{60q}", sections["timestep-change"])
        self.assertIn("alpha=0.01", sections["timestep-change"])
        self.assertIn("0.03", sections["timestep-change"])
        self.assertIn("稳定步长", sections["parameter-mapping"])
        self.assertIn("质量弹簧使用力积分", sections["iteration-substeps"])
        self.assertIn("零柔度", sections["parameter-mapping"])
        self.assertIn("浏览器概念实验", sections["pseudocode-debugging"])
        self.assertIn("当前 Python 实现", sections["pseudocode-debugging"])
        self.assertIn("float32", sections["pseudocode-debugging"])
        text += "".join(demo.description for demo in row.demos)
        for marker in (r"springForce(1,1.2,120,0)=-120", r"v_a'=1.4", r"v_b'=-1.4",
                       r"L'=3-2.8H\approx2.953333\,\mathrm m", r"K'=1.96\,\mathrm J",
                       r"U'\approx54.530667\,\mathrm J", "stableSpringStep(k, m/2)",
                       "所选与参考绝对长度差"):
            self.assertIn(marker, text, marker)
        for stale in ("3.086667", "3.08667", "70.850667", "70.851", "所选减参考长度差",
                      "stableSpringStep(k, m)"):
            self.assertNotIn(stale, text, stale)
        source_paths = {source.path for source in row.sources}
        self.assertEqual(source_paths, {
            "physics/cloth/mass_spring.py", "physics/cloth/pbd.py", "physics/cloth/xpbd.py",
            "physics/common/simulation.py", "book/assets/physics-core.js",
        })
        for source in row.sources:
            self.assertTrue((REPO_DIR / source.path).is_file(), source.path)
            self.assertIn("/blob/feat/m1-qt-minimal-loop/", source.url)


class PhysicsChapterNineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tempdir.cleanup)
        cls.output = Path(cls.tempdir.name)
        physics_build.build(cls.output)
        cls.rows = {row.number: row for row in physics_content.chapters()}

    def test_chapter_nine_has_twelve_sections_and_exact_demo_bindings(self):
        self.assertIn(9, self.rows, "chapter 9 must be available")
        row = self.rows[9]
        self.assertEqual((row.slug, row.unit, row.title),
                         ("physics09", "单元四：从点到刚体和软体", "刚体、旋转与地面接触"))
        self.assertEqual(len(row.sections), 12)
        self.assertEqual([section.id for section in row.sections], [
            "particle-rigid-body", "translation", "angular-velocity", "moment-of-inertia",
            "torque", "sphere-collision", "ground-contact", "rolling-sliding", "friction",
            "sleeping-jitter", "current-boundary", "worked-example",
        ])
        for section in row.sections:
            self.assertGreater(len(section.html), 500, section.id)
            self.assertIn("<p>", section.html)
            self.assertIn("</p>", section.html)
            self.assertNotIn("$", section.html)
        self.assertEqual([demo.id for demo in row.demos], ["rigid-contact", "rotation-intuition"])
        self.assertEqual([demo.section_id for demo in row.demos],
                         ["ground-contact", "angular-velocity"])
        self.assertEqual([demo.script for demo in row.demos],
                         ["demos/physics17.js", "demos/physics18.js"])
        ids = [section.id for section in row.sections] + [demo.id for demo in row.demos]
        self.assertEqual(len(ids), len(set(ids)))
        text = (self.output / "physics/chapters/physics09.html").read_text(encoding="utf-8")
        for number in range(1, 13):
            self.assertIn(f"9.{number}", text)
        tree = HTMLTree(text)
        for demo_id, section_id in (("rigid-contact", "ground-contact"),
                                    ("rotation-intuition", "angular-velocity")):
            demo = next(element for element in tree.elements if element["attrs"].get("id") == demo_id)
            section = next(element for element in tree.elements
                           if element["attrs"].get("id") == demo["parent"]["attrs"].get("id"))
            self.assertEqual(section["attrs"]["id"], section_id)
            self.assertEqual([child["attrs"].get("class") for child in tree.elements
                              if child["parent"] is section],
                             ["section-label", "prose", "demo-mount physics-demo"])

    def test_chapter_nine_separates_python_translation_from_browser_rotation_concepts(self):
        row = self.rows[9]
        sections = {section.id: section.html for section in row.sections}
        text = "".join(section.html for section in row.sections)
        for marker in ("质点", "刚体", "平移", "角速度", "转动惯量", "力矩", "恢复系数",
                       "球碰撞", "地面接触", "滚动", "滑动", "摩擦", "休眠", "抖动",
                       "当前边界", "算例", "调试"):
            self.assertIn(marker, text, marker)
        for marker in ("SphereBody", "RigidBodySimulation", "只有平移", "没有角速度",
                       "没有转动惯量", "没有力矩", "没有滚动", "动态刚体对", "CCD",
                       "float32", "双精度", "不逐位相同"):
            self.assertIn(marker, sections["current-boundary"], marker)
        self.assertIn("multiplicative", sections["current-boundary"])
        self.assertIn("切向阻尼", sections["current-boundary"])
        self.assertIn("静态平面", sections["current-boundary"])
        self.assertIn("静态球", sections["current-boundary"])
        self.assertIn("惯性张量", sections["current-boundary"])
        self.assertIn("接触流形", sections["current-boundary"])
        self.assertIn("浏览器概念实验", sections["current-boundary"])
        self.assertIn("不是生产引擎", sections["current-boundary"])
        self.assertIn(r"\(\omega=\frac{\Delta\theta}{\Delta t}\)", sections["angular-velocity"])
        self.assertIn(r"\(I=\frac12mR^2\)", sections["moment-of-inertia"])
        self.assertIn(r"\(\tau=FR\)", sections["torque"])
        self.assertIn(r"\(I_{\mathrm{eff}}=I+mR^2\)", sections["angular-velocity"])
        self.assertIn(r"\(\alpha=\frac{\tau}{I+mR^2}\)", sections["angular-velocity"])
        self.assertIn(r"\(\theta=\frac12\alpha t^2\)", sections["angular-velocity"])
        self.assertIn(r"\(x=R\theta\)", sections["rolling-sliding"])
        self.assertIn("1/120", sections["ground-contact"])
        self.assertIn("1/60", sections["ground-contact"])
        self.assertIn("6", sections["ground-contact"])
        self.assertIn("0.35", sections["ground-contact"])
        self.assertIn("0.58", sections["ground-contact"])
        self.assertIn("0.18", sections["ground-contact"])
        self.assertIn("1.8", sections["ground-contact"])
        self.assertIn("2.35", sections["ground-contact"])
        self.assertIn("碰撞前", sections["ground-contact"])
        self.assertIn(r"\(vy<0\)", sections["ground-contact"])
        self.assertIn("球心最高高度", sections["ground-contact"])
        self.assertIn("绝对水平位移", sections["ground-contact"])
        self.assertIn("m=1", sections["angular-velocity"])
        self.assertIn("R=0.5", sections["angular-velocity"])
        self.assertIn("实心圆盘", sections["moment-of-inertia"])
        self.assertIn("质量分布", sections["moment-of-inertia"])
        self.assertIn("有限时间范围", sections["angular-velocity"])
        self.assertIn("理想无滑动", sections["angular-velocity"])
        source_paths = {source.path for source in row.sources}
        self.assertEqual(source_paths, {
            "physics/rigidbody/sphere.py", "physics/common/simulation.py",
            "physics/common/colliders.py", "book/assets/physics-core.js",
        })
        for source in row.sources:
            self.assertTrue((REPO_DIR / source.path).is_file(), source.path)
            self.assertIn("/blob/feat/m1-qt-minimal-loop/", source.url)
            self.assertTrue(source.status)

    def test_chapter_nine_formulas_render_strictly_with_local_katex(self):
        text = (self.output / "physics/chapters/physics09.html").read_text(encoding="utf-8")
        formulas = [
            {"tex": html.unescape(tex), "display": True}
            for tex in re.findall(r"\\\[(.*?)\\\]", text, re.DOTALL)
        ]
        formulas.extend(
            {"tex": html.unescape(tex), "display": False}
            for tex in re.findall(r"\\\((.*?)\\\)", text, re.DOTALL)
        )
        self.assertGreaterEqual(len(formulas), 180)
        renderer = REPO_DIR / "book/assets/vendor/katex/katex.min.js"
        result = subprocess.run([
            "node", "-e", textwrap.dedent("""
                const fs = require('node:fs');
                const katex = require(process.argv[1]);
                const formulas = JSON.parse(fs.readFileSync(0, 'utf8'));
                formulas.forEach((formula, index) => {
                  try {
                    katex.renderToString(formula.tex, {
                      displayMode: formula.display, throwOnError: true, strict: 'error',
                    });
                  } catch (error) {
                    console.error(`formula ${index}: ${error.message}: ${formula.tex}`);
                    process.exit(1);
                  }
                });
                console.log(`Rendered ${formulas.length} LaTeX formulas`);
            """), str(renderer),
        ], input=json.dumps(formulas), text=True, capture_output=True, timeout=10, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertIn(f"Rendered {len(formulas)} LaTeX formulas", result.stdout)


class PhysicsChapterDemoTests(unittest.TestCase):
    def test_demo_scripts_use_iterable_conversion_for_dom_children(self):
        invalid = []
        for script in sorted((REPO_DIR / "book/assets/demos").glob("physics*.js")):
            for number, line in enumerate(script.read_text(encoding="utf-8").splitlines(), 1):
                if re.search(r"\.children\.(?:find|filter|map)\(", line):
                    invalid.append(f"{script.name}:{number}")
                if re.search(r"\.tagName\s*={2,3}\s*['\"][a-z]", line):
                    invalid.append(f"{script.name}:{number}")
        self.assertEqual(invalid, [], "Real DOM collections/tag names differ from the fake DOM: " + ", ".join(invalid))

    def run_demos(self, scripts, body, setup=""):
        harness = textwrap.dedent(r"""
            const assert = require('node:assert/strict');
            const fs = require('node:fs');
            const vm = require('node:vm');
            class Element {
              constructor(tag) {
                this.tagName = tag; this.children = []; this.attributes = {}; this.listeners = {};
                this.className = ''; this.textContent = ''; this.value = ''; this.captured = [];
                this.textDraws = []; this.pathDraws = []; this.contextFont = '';
              }
              appendChild(child) { this.children.push(child); child.parent = this; return child; }
              append(...children) { children.forEach(child => this.appendChild(child)); }
              replaceChildren() { this.children = []; }
              setAttribute(key, value) { this.attributes[key] = value; }
              addEventListener(key, fn) { (this.listeners[key] ||= []).push(fn); }
              dispatch(key, extra = {}) {
                const event = Object.assign({target: this}, extra);
                for (const fn of this.listeners[key] || []) fn(event);
                if (this.parent) this.parent.dispatch(key, event);
              }
              setPointerCapture(id) { this.captured.push(id); }
              getContext() {
                return new Proxy({}, {
                  get: (_, key) => {
                    if (key === 'font') return this.contextFont;
                    if (key === 'fillText') return (text, x, y) => {
                      this.textDraws.push({text, x, y, font: this.contextFont});
                    };
                    if (key === 'measureText') return text => ({width: text.length * 13});
                    if (key === 'moveTo' || key === 'lineTo') return (x, y) => {
                      this.pathDraws.push({op: key, x, y});
                    };
                    if (key === 'arc') return (x, y, r, start, end) =>
                      this.pathDraws.push({op: key, x, y, r, start, end});
                    return () => {};
                  },
                  set: (_, key, value) => {
                    if (key === 'font') this.contextFont = value;
                    return true;
                  },
                });
              }
              getBoundingClientRect() { return {left: 0, top: 0, width: this.clientWidth || 320,
                height: this.clientHeight || 180}; }
            }
            const roots = {
              'particle-lab': new Element('div'), 'frame-states': new Element('div'),
              'integration-race': new Element('div'), 'fixed-step': new Element('div'),
              'force-balance': new Element('div'), 'energy-ledger': new Element('div'),
              'spring-oscillator': new Element('div'), 'spring-cloth': new Element('div'),
              'pbd-rope': new Element('div'), 'constraint-order': new Element('div'),
              'collision-friction': new Element('div'), 'continuous-collision': new Element('div'),
              'xpbd-compliance': new Element('div'), 'area-constraint': new Element('div'),
              'solver-race': new Element('div'), 'substep-invariance': new Element('div'),
              'rigid-contact': new Element('div'), 'rotation-intuition': new Element('div'),
              'cloth-showcase': new Element('div'), 'soft-area': new Element('div')
            };
            Object.entries(roots).forEach(([id, root]) => { root.id = id; });
            const pending = new Map();
            const requested = [];
            const cancelled = [];
            let nextFrameId = 1;
            const window = {devicePixelRatio: 1, matchMedia: () => ({matches: false}),
              requestAnimationFrame: callback => {
                const id = nextFrameId++; pending.set(id, callback); requested.push(id); return id;
              },
              cancelAnimationFrame: id => { cancelled.push(id); pending.delete(id); }};
            window.Element = Element;
            window.addEventListener = () => {};
            window.document = {getElementById: id => roots[id], createElement: tag => new Element(tag)};
            const document = window.document;
            const context = vm.createContext({window, document: window.document, console});
            const paths = process.argv.slice(1);
            for (const path of paths.slice(0, 2)) {
              vm.runInContext(fs.readFileSync(path, 'utf8'), context, {filename: path});
            }
            const mounts = [];
            const mount = window.RTPhysicsUI.mount;
            window.RTPhysicsUI.mount = (...args) => {
              const controller = mount(...args); mounts.push(controller); return controller;
            };
            __DEMO_SETUP__
            for (const path of paths.slice(2)) {
              vm.runInContext(fs.readFileSync(path, 'utf8'), context, {filename: path});
            }
            const children = element => element.children.flatMap(child => [child, ...children(child)]);
            const inputs = root => children(root).filter(element => element.tagName === 'input');
            const readout = root => children(root).find(element => element.attributes['data-readout'] !== undefined);
            function tick(time) {
              const entry = pending.entries().next().value;
              if (!entry) return false;
              pending.delete(entry[0]);
              entry[1](time);
              return true;
            }
        """).replace("__DEMO_SETUP__", textwrap.dedent(setup))
        result = subprocess.run(
            ["node", "-e", harness + "(() => {\n" + textwrap.dedent(body) + "\n})();",
             str(CORE_PATH), str(UI_PATH), *map(str, scripts)],
            cwd=REPO_DIR, capture_output=True, text=True, timeout=10, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        self.assertEqual(result.stderr, "")
        return json.loads(result.stdout)

    def test_particle_lab_starts_paused_without_an_active_frame(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics01.js"], """
            const assert = require('node:assert/strict');
            assert.equal(mounts.length, 1);
            const root = document.getElementById('particle-lab');
            const toolbar = root.children.find(element => element.className === 'demo-toolbar physics-controls');
            assert.equal(toolbar.children[1].textContent, '运行');
            assert.equal(pending.size, 0);
            assert.equal(mounts[0].getState().frame, 0);
            console.log(JSON.stringify({paused: true, pending: pending.size}));
        """)
        self.assertEqual(data, {"paused": True, "pending": 0})

    def test_particle_lab_pauses_before_leaving_inclusive_bounds_and_guides_reset_or_drag(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics01.js"], """
            const assert = require('node:assert/strict');
            const root = document.getElementById('particle-lab');
            const lab = mounts[0];
            const toolbar = root.children.find(element => element.className === 'demo-toolbar physics-controls');
            lab.setRunning(true);
            tick(0);
            tick(1000);
            assert.deepEqual(JSON.parse(JSON.stringify(lab.getState().p)), {x: 6, y: -0.5});
            assert.equal(lab.getState().frame, 5);
            assert.equal(pending.size, 0);
            assert.equal(toolbar.children[1].textContent, '运行');
            const warning = children(root).find(element => element.attributes.role === 'status');
            assert.ok(warning);
            assert.equal(warning.hidden, false);
            assert.match(warning.textContent, /重置/);
            assert.match(warning.textContent, /拖拽/);
            assert.match(warning.textContent, /边界|范围/);
            toolbar.children[0].dispatch('click');
            assert.deepEqual(JSON.parse(JSON.stringify(lab.getState().p)), {x: 1, y: 2});
            assert.equal(lab.getState().frame, 0);
            assert.equal(warning.hidden, true);
            assert.equal(warning.textContent, '');
            assert.equal(pending.size, 0);

            const controls = inputs(root);
            controls[0].value = '0'; controls[0].dispatch('input');
            controls[1].value = '1'; controls[1].dispatch('input');
            controls[2].value = '1'; controls[2].dispatch('input');
            lab.step();
            lab.step();
            lab.step();
            lab.step();
            assert.deepEqual(JSON.parse(JSON.stringify(lab.getState().p)), {x: 1, y: 5});
            lab.step();
            assert.deepEqual(JSON.parse(JSON.stringify(lab.getState().p)), {x: 1, y: 5});
            assert.equal(lab.getState().frame, 3);
            assert.equal(warning.hidden, false);
            console.log(JSON.stringify({pausedAtBoundary: true, guided: true, reset: true}));
        """)
        self.assertEqual(data, {"pausedAtBoundary": True, "guided": True, "reset": True})

    def test_particle_lab_prediction_drag_reset_and_finite_readouts(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics01.js"], """
            const assert = require('node:assert/strict');
            assert.equal(mounts.length, 1);
            const lab = mounts[0];
            const snapshot = () => JSON.parse(JSON.stringify(lab.getState()));
            assert.deepEqual(snapshot(), {
              p: {x: 1, y: 2}, v: {x: 2, y: -1}, a: {x: 0, y: 0}, dt: 0.5, frame: 0
            });
            const controls = inputs(document.getElementById('particle-lab'));
            controls[0].value = '4'; controls[0].dispatch('input');
            controls[1].value = '-2'; controls[1].dispatch('input');
            controls[2].value = '0.25'; controls[2].dispatch('input');
            assert.deepEqual(snapshot().v, {x: 4, y: -2});
            assert.match(readout(document.getElementById('particle-lab')).textContent,
                         /下一帧位置 x.*2.*下一帧位置 y.*1\.5/);
            lab.step();
            assert.deepEqual(snapshot().p, {x: 2, y: 1.5});
            lab.reset();
            const canvas = lab.canvas;
            const start = {x: 45 + (2 / 7) * 255, y: 24 + (3 / 6) * 121};
            const target = {x: 45 + (4 / 7) * 255, y: 24 + (2 / 6) * 121};
            canvas.dispatch('pointerdown', {clientX: start.x, clientY: start.y, pointerId: 7});
            canvas.dispatch('pointermove', {clientX: target.x, clientY: target.y, pointerId: 7});
            canvas.dispatch('pointerup', {pointerId: 7});
            assert.ok(Math.abs(snapshot().p.x - 3) < 1e-12);
            assert.ok(Math.abs(snapshot().p.y - 3) < 1e-12);
            lab.reset();
            assert.deepEqual(snapshot(), {
              p: {x: 1, y: 2}, v: {x: 2, y: -1}, a: {x: 0, y: 0}, dt: 0.5, frame: 0
            });
            assert.doesNotMatch(readout(document.getElementById('particle-lab')).textContent, /NaN|Infinity/);
            assert.match(readout(document.getElementById('particle-lab')).textContent, /2\.236/);
            console.log(JSON.stringify({ok: true, dragged: true}));
        """)
        self.assertEqual(data, {"ok": True, "dragged": True})

    def test_frame_states_legend_rows_fit_narrow_canvas_without_overlap(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics02.js"], """
            const assert = require('node:assert/strict');
            const frames = mounts[0];
            frames.setRunning(false);
            frames.canvas.clientWidth = 328;
            frames.canvas.textDraws = [];
            frames.draw();
            const rows = frames.canvas.textDraws.filter(call => call.text.startsWith('■') || call.text.startsWith('○'));
            assert.equal(rows.length, 2);
            for (const row of rows) {
              assert.ok(row.x >= 0);
              assert.ok(row.x + row.text.length * 13 <= 328, row.text + ' exceeds the 328px canvas');
            }
            assert.ok(rows[1].y - rows[0].y >= 13, 'legend rows overlap vertically');
            console.log(JSON.stringify({rows, width: 328}));
        """)
        self.assertEqual([row["text"] for row in data["rows"]], [
            "■ 实际帧位置（方块/实心点）",
            "○ 线性插值位置；虚线为两者距离",
        ])
        self.assertEqual(data["width"], 328)

    def test_frame_states_controls_interpolation_and_reset(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics02.js"], """
            const assert = require('node:assert/strict');
            assert.equal(mounts.length, 1);
            const frames = mounts[0];
            frames.setRunning(false);
            const root = document.getElementById('frame-states');
            const table = root.children.find(element => element.tagName === 'table');
            const tbody = table.children.find(element => element.tagName === 'tbody');
            assert.equal(tbody.children.length, 5);
            const controls = inputs(root);
            controls[0].value = '60'; controls[0].dispatch('input');
            controls[1].value = '2'; controls[1].dispatch('input');
            frames.step();
            let state = frames.getState();
            assert.equal(state.frame, 2);
            assert.ok(Math.abs(state.time - 1 / 30) < 1e-12);
            assert.deepEqual(JSON.parse(JSON.stringify(state.actual)), {x: 1 + 2 / 30, y: 2 - 1 / 30});
            assert.equal(tbody.children[2].children[1].textContent, '0.033');
            assert.ok(state.error < 1e-12);
            frames.reset();
            state = frames.getState();
            assert.equal(state.time, 0);
            assert.equal(state.frame, 0);
            controls[0].value = '10'; controls[0].dispatch('input');
            controls[1].value = '0.5'; controls[1].dispatch('input');
            frames.step();
            state = frames.getState();
            assert.ok(Math.abs(state.time - 1 / 120) < 1e-12);
            assert.equal(state.frame, 0);
            assert.ok(state.error > 0);
            assert.ok(Math.abs(state.interpolated.x - 1 - 2 / 120) < 1e-12);
            assert.ok(Math.abs(state.interpolated.y - 2 + 1 / 120) < 1e-12);
            assert.equal(tbody.children.length, 5);
            assert.doesNotMatch(readout(root).textContent, /NaN|Infinity/);
            assert.match(readout(root).textContent, /插值误差/);
            frames.reset();
            assert.equal(frames.getState().time, 0);
            assert.equal(Number(controls[0].value), 10);
            assert.equal(Number(controls[1].value), 1);
            console.log(JSON.stringify({ok: true, interpolated: true}));
        """)
        self.assertEqual(data, {"ok": True, "interpolated": True})

    def test_integration_race_starts_paused_and_compares_three_real_integrators(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics03.js"], """
            const assert = require('node:assert/strict');
            assert.equal(mounts.length, 1);
            const root = document.getElementById('integration-race');
            const toolbar = root.children.find(element => element.className === 'demo-toolbar physics-controls');
            const race = mounts[0];
            assert.equal(toolbar.children[1].textContent, '运行');
            assert.equal(pending.size, 0);
            assert.equal(race.getState().frame, 0);
            assert.equal(race.getState().time, 0);
            const controls = inputs(root);
            assert.deepEqual(controls.map(input => Number(input.value)), [0.1, 20, -9.81]);
            race.step();
            const state = race.getState();
            assert.equal(state.frame, 1);
            assert.ok(Math.abs(state.time - 0.1) < 1e-12);
            assert.ok(Math.abs(state.current.forward.y - 2) < 1e-12);
            assert.ok(Math.abs(state.current['semi-implicit'].y - 1.9019) < 1e-12);
            assert.ok(Math.abs(state.current.exact.y - 1.95095) < 1e-12);
            assert.ok(Math.abs(state.current.exact.vy - 19.019) < 1e-12);
            assert.ok(exactResults.length > state.plannedSteps,
                      exactResults.length + ' shared exact calls for ' + state.plannedSteps + ' planned steps');
            assert.strictEqual(state.current.exact, exactResults.at(-1));
            assert.equal(state.paths.forward.length, state.plannedSteps + 1);
            assert.equal(state.paths['semi-implicit'].length, state.plannedSteps + 1);
            assert.equal(state.paths.exact.length, state.plannedSteps + 1);
            assert.equal(state.trails.forward.length, 2);
            assert.equal(state.trails['semi-implicit'].length, 2);
            assert.equal(state.trails.exact.length, 2);
            assert.ok(Math.abs(state.endpoints.exact.y) < 1e-10);
            assert.ok(Math.abs(state.endpoints.forward.y - 1.9914393476044676) < 1e-10);
            assert.ok(Math.abs(state.endpoints['semi-implicit'].y + 1.991439347604503) < 1e-10);
            assert.ok(Math.abs(state.errors.forward - 1.9914393476044827) < 1e-10);
            assert.ok(Math.abs(state.errors['semi-implicit'] - 1.9914393476044878) < 1e-10);
            const readoutText = readout(root).textContent;
            assert.match(readoutText, /终点/);
            assert.match(readoutText, /前向欧拉误差/);
            assert.match(readoutText, /半隐式欧拉误差/);
            assert.doesNotMatch(readoutText, /NaN|Infinity/);
            console.log(JSON.stringify({oneStep: state.current, endpoint: state.endpoints.exact}));
        """, setup="""
            const sharedIntegrate = window.RTPhysics.integrateProjectile;
            const exactResults = [];
            window.RTPhysics.integrateProjectile = (...args) => {
              const next = sharedIntegrate(...args);
              if (args[3] === 'exact') exactResults.push(next);
              return next;
            };
        """)
        self.assertEqual(data["oneStep"]["forward"]["y"], 2)
        self.assertAlmostEqual(data["oneStep"]["exact"]["y"], 1.95095)
        self.assertAlmostEqual(data["endpoint"]["y"], 0)

    def test_integration_race_controls_bound_playback_and_reset_deterministically(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics03.js"], """
            const assert = require('node:assert/strict');
            const root = document.getElementById('integration-race');
            const race = mounts[0];
            const controls = inputs(root);
            controls[0].value = '0.2'; controls[0].dispatch('input');
            controls[1].value = '10'; controls[1].dispatch('input');
            controls[2].value = '-5'; controls[2].dispatch('input');
            let state = race.getState();
            assert.equal(state.dt, 0.2);
            assert.equal(state.v0, 10);
            assert.equal(state.gravity, -5);
            assert.equal(state.frame, 0);
            assert.ok(Math.abs(state.duration - 4) < 1e-12);
            race.setRunning(true);
            tick(0);
            tick(1000);
            tick(1250); // Shared transport caps each caught-up frame at 0.25 s.
            state = race.getState();
            assert.equal(state.frame, 20);
            assert.equal(state.finished, true);
            assert.equal(pending.size, 0);
            const warning = children(root).find(element => element.attributes.role === 'status');
            assert.equal(warning.hidden, false);
            assert.match(warning.textContent, /终点/);
            assert.match(warning.textContent, /重置/);
            const finished = JSON.parse(JSON.stringify(state));
            race.step();
            assert.deepEqual(JSON.parse(JSON.stringify(race.getState())), finished);
            race.reset();
            state = race.getState();
            assert.equal(state.frame, 0);
            assert.equal(state.time, 0);
            assert.equal(state.finished, false);
            assert.deepEqual(controls.map(input => Number(input.value)), [0.1, 20, -9.81]);
            assert.equal(warning.hidden, true);
            assert.equal(pending.size, 0);
            assert.doesNotMatch(readout(root).textContent, /NaN|Infinity/);
            console.log(JSON.stringify({frame: finished.frame, reset: true}));
        """)
        self.assertEqual(data, {"frame": 20, "reset": True})

    def test_integration_race_three_row_legend_fits_narrow_canvas(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics03.js"], """
            const assert = require('node:assert/strict');
            const race = mounts[0];
            race.setRunning(false);
            race.canvas.clientWidth = 328;
            race.canvas.textDraws = [];
            race.draw();
            const rows = race.canvas.textDraws.filter(call => call.text.startsWith('■') || call.text.startsWith('○'));
            assert.equal(rows.length, 3);
            for (const row of rows) {
              assert.ok(row.x >= 0);
              assert.ok(row.x + row.text.length * 13 <= 328, row.text + ' exceeds the 328px canvas');
            }
            for (let index = 1; index < rows.length; index += 1) {
              assert.ok(rows[index].y - rows[index - 1].y >= 13, 'legend rows overlap vertically');
            }
            console.log(JSON.stringify({rows, width: 328}));
        """)
        self.assertEqual([row["text"] for row in data["rows"]],
                         ["■ 前向欧拉", "■ 半隐式欧拉", "○ 精确解（参考）"])
        self.assertEqual(data["width"], 328)

    def test_fixed_step_compares_direct_dt_with_accumulator_and_resets(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics04.js"], """
            const assert = require('node:assert/strict');
            assert.equal(mounts.length, 1);
            const root = document.getElementById('fixed-step');
            const toolbar = root.children.find(element => element.className === 'demo-toolbar physics-controls');
            const fixed = mounts[0];
            assert.equal(toolbar.children[1].textContent, '运行');
            assert.equal(pending.size, 0);
            const controls = inputs(root);
            assert.deepEqual(controls.map(input => Number(input.value)), [60, 120, 120]);
            for (let index = 0; index < 4; index += 1) fixed.step();
            let state = fixed.getState();
            assert.equal(state.frame, 4);
            assert.equal(state.substeps, 8);
            assert.ok(Math.abs(state.fixedTime - 1 / 15) < 1e-12);
            assert.ok(Math.abs(state.directTime - 1 / 15) < 1e-12);
            fixed.step();
            state = fixed.getState();
            assert.equal(state.frame, 5);
            assert.equal(state.substeps, 24);
            assert.ok(Math.abs(state.elapsed - 61 / 300) < 1e-12);
            assert.ok(Math.abs(state.fixedTime - 0.2) < 1e-12);
            assert.equal(state.directTime, state.elapsed);
            assert.ok(Math.abs(state.maxCatchupError - 1 / 300) < 1e-12);
            assert.ok(Math.abs(state.fixed.y - (24 / 120 * 20 - 9.81 / 120 ** 2 * 24 * 25 / 2)) < 1e-12);
            let directY = 0;
            let directVy = 20;
            for (let index = 0; index < 4; index += 1) {
              directVy += -9.81 / 60;
              directY += directVy / 60;
            }
            const laggedFrameDt = 1 / 60 + 0.12;
            directVy += -9.81 * laggedFrameDt;
            directY += directVy * laggedFrameDt;
            assert.ok(Math.abs(state.direct.y - directY) < 1e-12);
            // These positions are at different times after the stalled frame;
            // compare elapsed versus fixed simulated time, not raw heights.
            assert.ok(state.fixedTime < state.directTime);
            assert.ok(Math.abs(state.fixed.x - 5 * state.fixedTime) < 1e-12);
            const readoutText = readout(root).textContent;
            assert.match(readoutText, /模拟时间/);
            assert.match(readoutText, /子步数/);
            assert.match(readoutText, /最大追帧误差/);
            assert.doesNotMatch(readoutText, /NaN|Infinity/);
            fixed.reset();
            state = fixed.getState();
            assert.equal(state.frame, 0);
            assert.equal(state.elapsed, 0);
            assert.equal(state.substeps, 0);
            assert.deepEqual(controls.map(input => Number(input.value)), [60, 120, 120]);
            console.log(JSON.stringify({frame: 5, fixedY: state.fixed.y, reset: true}));
        """)
        self.assertEqual(data["frame"], 5)
        self.assertAlmostEqual(data["fixedY"], 0)
        self.assertTrue(data["reset"])

    def test_fixed_step_changes_controls_stops_at_bound_and_keeps_paused_single_step(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics04.js"], """
            const assert = require('node:assert/strict');
            const root = document.getElementById('fixed-step');
            const fixed = mounts[0];
            const controls = inputs(root);
            controls[0].value = '30'; controls[0].dispatch('input');
            controls[1].value = '60'; controls[1].dispatch('input');
            controls[2].value = '200'; controls[2].dispatch('input');
            fixed.step();
            let state = fixed.getState();
            assert.equal(state.frame, 1);
            assert.ok(Math.abs(state.elapsed - 1 / 30) < 1e-12);
            assert.equal(state.substeps, 2);
            assert.ok(Math.abs(state.fixedTime - 1 / 30) < 1e-12);
            for (let index = 0; index < 4; index += 1) fixed.step();
            state = fixed.getState();
            assert.equal(state.frame, 5);
            assert.ok(Math.abs(state.elapsed - (5 / 30 + 0.2)) < 1e-12);
            assert.equal(state.substeps, 22);
            assert.ok(Math.abs(state.fixedTime - 22 / 60) < 1e-12);
            fixed.reset();
            fixed.setRunning(true);
            tick(0);
            tick(1000);
            state = fixed.getState();
            assert.equal(state.frame, 12);
            assert.equal(state.finished, true);
            assert.equal(pending.size, 0);
            const warning = children(root).find(element => element.attributes.role === 'status');
            assert.equal(warning.hidden, false);
            assert.match(warning.textContent, /12/);
            assert.match(warning.textContent, /重置/);
            const finished = JSON.parse(JSON.stringify(state));
            fixed.step();
            assert.deepEqual(JSON.parse(JSON.stringify(fixed.getState())), finished);
            fixed.reset();
            fixed.setRunning(false);
            fixed.step();
            assert.equal(fixed.getState().frame, 1);
            console.log(JSON.stringify({bounded: true, pausedStep: true}));
        """)
        self.assertEqual(data, {"bounded": True, "pausedStep": True})

    def test_fixed_step_side_by_side_labels_fit_narrow_canvas(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics04.js"], """
            const assert = require('node:assert/strict');
            const fixed = mounts[0];
            fixed.canvas.clientWidth = 328;
            fixed.canvas.textDraws = [];
            fixed.draw();
            const headings = fixed.canvas.textDraws.filter(call => call.text === '每帧直接 dt' || call.text === '累加器固定步长');
            assert.equal(headings.length, 2);
            for (const heading of headings) {
              assert.ok(heading.x >= 0);
              assert.ok(heading.x + heading.text.length * 13 <= 328, heading.text + ' exceeds 328px');
            }
            const verticalLabels = fixed.canvas.textDraws.filter(call => call.text === 'y ↑' || call.text === 'y ↓');
            assert.deepEqual(verticalLabels.map(call => call.text), ['y ↑', 'y ↑']);
            console.log(JSON.stringify({headings, verticalLabels, width: 328}));
        """)
        self.assertEqual(data["width"], 328)
        self.assertEqual([heading["text"] for heading in data["headings"]],
                         ["每帧直接 dt", "累加器固定步长"])

    def test_force_balance_defaults_pause_and_integrate_downward_forces(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics05.js"], """
            const assert = require('node:assert/strict');
            assert.equal(mounts.length, 1);
            const root = document.getElementById('force-balance');
            const toolbar = root.children.find(element => element.className === 'demo-toolbar physics-controls');
            const force = mounts[0];
            assert.equal(toolbar.children[1].textContent, '运行');
            assert.equal(pending.size, 0);
            assert.deepEqual(inputs(root).map(input => Number(input.value)), [1, 2, 0]);
            let state = force.getState();
            assert.equal(state.terminalSpeed, 4.905);
            assert.equal(state.velocity, 0);
            assert.equal(state.netForce, 9.81);
            assert.equal(state.ball.w, 1);
            assert.match(force.canvas.attributes['aria-label'], /重力.*阻力.*合力/);
            force.step();
            state = force.getState();
            assert.equal(state.frame, 1);
            assert.ok(Math.abs(state.time - 1 / 60) < 1e-12);
            assert.ok(Math.abs(state.velocity - 9.81 / 60) < 1e-12);
            assert.ok(Math.abs(state.distance - 9.81 / 3600) < 1e-12);
            assert.ok(Math.abs(state.dragForce - 19.62 / 60) < 1e-12);
            assert.ok(Math.abs(state.netForce - 9.81 + 19.62 / 60) < 1e-12);
            assert.ok(Math.abs(state.acceleration - state.netForce) < 1e-12);
            assert.ok(Math.abs(state.ball.vy + state.velocity) < 1e-12);
            const text = readout(root).textContent;
            for (const marker of ['合力', '加速度', '当前速度', '终端速度']) assert.ok(text.includes(marker));
            assert.doesNotMatch(text, /NaN|Infinity/);
            console.log(JSON.stringify({velocity: state.velocity, terminal: state.terminalSpeed}));
        """)
        self.assertAlmostEqual(data["velocity"], 0.1635)
        self.assertEqual(data["terminal"], 4.905)

    def test_force_balance_buoyancy_uses_net_weight_and_resets_defaults(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics05.js"], """
            const assert = require('node:assert/strict');
            const root = document.getElementById('force-balance');
            const force = mounts[0];
            const controls = inputs(root);
            controls[2].value = '1'; controls[2].dispatch('input');
            let state = force.getState();
            assert.equal(state.buoyancy, 1);
            assert.equal(state.buoyantForce, 4);
            assert.ok(Math.abs(state.terminalSpeed - 2.905) < 1e-12);
            assert.ok(force.canvas.textDraws.some(call => call.text.includes('浮力开启：v_t=(mg-F_b)/c')));
            controls[0].value = '2'; controls[0].dispatch('input');
            controls[1].value = '1'; controls[1].dispatch('input');
            state = force.getState();
            assert.ok(Math.abs(state.terminalSpeed - 15.62) < 1e-12);
            assert.equal(state.time, 0);
            assert.equal(state.velocity, 0);
            force.step();
            assert.ok(Math.abs(force.getState().velocity - 15.62 / 120) < 1e-12);
            force.reset();
            state = force.getState();
            assert.deepEqual(controls.map(input => Number(input.value)), [1, 2, 0]);
            assert.equal(state.terminalSpeed, 4.905);
            assert.equal(state.buoyantForce, 0);
            assert.ok(force.canvas.textDraws.some(call => call.text.includes('浮力关闭：v_t=mg/c')));
            console.log(JSON.stringify({buoyantTerminal: 2.905, resetTerminal: state.terminalSpeed}));
        """)
        self.assertEqual(data, {"buoyantTerminal": 2.905, "resetTerminal": 4.905})

    def test_force_balance_stops_at_bounded_observation_window(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics05.js"], """
            const assert = require('node:assert/strict');
            const root = document.getElementById('force-balance');
            const force = mounts[0];
            for (let index = 0; index < 720; index += 1) force.step();
            const state = force.getState();
            assert.equal(state.frame, 720);
            assert.equal(state.time, 12);
            assert.equal(state.finished, true);
            assert.equal(pending.size, 0);
            assert.ok(Number.isFinite(state.distance));
            assert.ok(Number.isFinite(state.ball.y));
            const warning = children(root).find(element => element.attributes.role === 'status');
            assert.equal(warning.hidden, false);
            assert.match(warning.textContent, /12/);
            assert.match(warning.textContent, /重置/);
            const finished = JSON.parse(JSON.stringify(state));
            force.step();
            assert.deepEqual(JSON.parse(JSON.stringify(force.getState())), finished);
            assert.doesNotMatch(readout(root).textContent, /NaN|Infinity/);
            console.log(JSON.stringify({frame: state.frame, distance: state.distance}));
        """)
        self.assertEqual(data["frame"], 720)
        self.assertTrue(data["distance"] > 0)

    def test_energy_ledger_draws_horizontal_one_dimensional_spring(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics06.js"], """
            const assert = require('node:assert/strict');
            const ledger = mounts[0];
            ledger.canvas.clientWidth = 328;
            ledger.canvas.clientHeight = 180;
            ledger.canvas.pathDraws = [];
            ledger.canvas.textDraws = [];
            ledger.draw();
            const start = ledger.canvas.pathDraws.findIndex(call => call.op === 'moveTo');
            assert.ok(start >= 0);
            const spring = ledger.canvas.pathDraws.slice(start, start + 9);
            assert.equal(spring.length, 9);
            assert.equal(spring[0].y, spring[8].y, 'spring endpoints must share a horizontal axis');
            assert.ok(spring.every(call => Math.abs(call.y - spring[0].y) <= 6), 'coil must stay near its horizontal axis');
            assert.ok(spring[8].x > spring[0].x, 'positive x must place the mass to the right');
            assert.ok(ledger.canvas.textDraws.some(call => call.text === '平衡位置 x=0'));
            console.log(JSON.stringify({springY: spring[0].y, horizontal: true}));
        """)
        self.assertEqual(data, {"springY": 58, "horizontal": True})

    def test_energy_ledger_uses_shared_core_and_hand_checked_first_step(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics06.js"], """
            const assert = require('node:assert/strict');
            assert.equal(mounts.length, 1);
            const root = document.getElementById('energy-ledger');
            const toolbar = root.children.find(element => element.className === 'demo-toolbar physics-controls');
            const energy = mounts[0];
            assert.equal(toolbar.children[1].textContent, '运行');
            assert.equal(pending.size, 0);
            assert.deepEqual(inputs(root).map(input => Number(input.value)), [1, 10, 0.5]);
            assert.match(energy.canvas.attributes['aria-label'], /弹簧.*动能.*势能.*总能量/);
            const springBefore = springCalls;
            const energyBefore = energyCalls;
            const stableBefore = stableStepCalls;
            energy.step();
            const state = energy.getState();
            const expectedTotal = 1 / 72 + 644405 / 129600;
            assert.equal(springCalls, springBefore + 1);
            assert.equal(energyCalls, energyBefore + 1);
            assert.equal(stableStepCalls, stableBefore);
            assert.equal(state.force, -10);
            assert.ok(Math.abs(state.v + 1 / 6) < 1e-12);
            assert.ok(Math.abs(state.x - 359 / 360) < 1e-12);
            assert.ok(Math.abs(state.kinetic - 1 / 72) < 1e-12);
            assert.ok(Math.abs(state.potential - 644405 / 129600) < 1e-12);
            assert.ok(Math.abs(state.totalEnergy - expectedTotal) < 1e-12);
            assert.ok(Math.abs(state.relativeEnergyLoss - (5 - expectedTotal) / 5) < 1e-12);
            assert.ok(Math.abs(state.stepHint - 0.6324555320336759) < 1e-12);
            const text = readout(root).textContent;
            for (const marker of ['动能', '势能', '总能量', '相对能量损失']) assert.ok(text.includes(marker));
            assert.doesNotMatch(text, /NaN|Infinity/);
            console.log(JSON.stringify({kinetic: state.kinetic, potential: state.potential,
              total: state.totalEnergy, loss: state.relativeEnergyLoss}));
        """, setup="""
            const sharedSpringForce = window.RTPhysics.springForce;
            const sharedEnergy = window.RTPhysics.energy1d;
            const sharedStableStep = window.RTPhysics.stableSpringStep;
            let springCalls = 0;
            let energyCalls = 0;
            let stableStepCalls = 0;
            window.RTPhysics.springForce = (...args) => {
              springCalls += 1;
              return sharedSpringForce(...args);
            };
            window.RTPhysics.energy1d = (...args) => {
              energyCalls += 1;
              return sharedEnergy(...args);
            };
            window.RTPhysics.stableSpringStep = (...args) => {
              stableStepCalls += 1;
              return sharedStableStep(...args);
            };
        """)
        self.assertAlmostEqual(data["kinetic"], 1 / 72)
        self.assertAlmostEqual(data["potential"], 644405 / 129600)
        self.assertGreater(data["loss"], 0)

    def test_energy_ledger_controls_keep_loss_finite_and_replay_deterministically(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics06.js"], """
            const assert = require('node:assert/strict');
            const root = document.getElementById('energy-ledger');
            const energy = mounts[0];
            const controls = inputs(root);
            controls[0].value = '0.5'; controls[0].dispatch('input');
            controls[1].value = '1'; controls[1].dispatch('input');
            controls[2].value = '5'; controls[2].dispatch('input');
            let state = energy.getState();
            assert.equal(state.initialEnergy, 0.5);
            assert.ok(Math.abs(state.stepHint - Math.SQRT2) < 1e-12);
            for (let index = 0; index < 10; index += 1) energy.step();
            state = energy.getState();
            for (const key of ['kinetic', 'potential', 'totalEnergy', 'relativeEnergyLoss']) {
              assert.ok(Number.isFinite(state[key]), key);
            }
            assert.ok(state.relativeEnergyLoss > 0);
            energy.reset();
            energy.step();
            const first = JSON.parse(JSON.stringify(energy.getState()));
            energy.reset();
            energy.step();
            assert.deepEqual(JSON.parse(JSON.stringify(energy.getState())), first);
            assert.deepEqual(controls.map(input => Number(input.value)), [1, 10, 0.5]);
            assert.doesNotMatch(readout(root).textContent, /NaN|Infinity/);
            console.log(JSON.stringify({finite: true, deterministic: true}));
        """)
        self.assertEqual(data, {"finite": True, "deterministic": True})

    def test_energy_ledger_stops_after_twelve_seconds_with_finite_loss(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics06.js"], """
            const assert = require('node:assert/strict');
            const root = document.getElementById('energy-ledger');
            const energy = mounts[0];
            for (let index = 0; index < 720; index += 1) energy.step();
            const state = energy.getState();
            assert.equal(state.frame, 720);
            assert.equal(state.time, 12);
            assert.equal(state.finished, true);
            assert.ok(state.relativeEnergyLoss > 0);
            assert.ok(state.relativeEnergyLoss < 1);
            assert.ok(Number.isFinite(state.totalEnergy));
            const warning = children(root).find(element => element.attributes.role === 'status');
            assert.equal(warning.hidden, false);
            assert.match(warning.textContent, /12/);
            const finished = JSON.parse(JSON.stringify(state));
            energy.step();
            assert.deepEqual(JSON.parse(JSON.stringify(energy.getState())), finished);
            assert.doesNotMatch(readout(root).textContent, /NaN|Infinity/);
            console.log(JSON.stringify({frame: state.frame, loss: state.relativeEnergyLoss}));
        """)
        self.assertEqual(data["frame"], 720)
        self.assertGreater(data["loss"], 0)
        self.assertLess(data["loss"], 1)

    def test_chapter_three_demo_labels_fit_narrow_canvas(self):
        data = self.run_demos([
            REPO_DIR / "book/assets/demos/physics05.js",
            REPO_DIR / "book/assets/demos/physics06.js",
        ], """
            const assert = require('node:assert/strict');
            const force = mounts[0];
            const energy = mounts[1];
            for (const controller of [force, energy]) {
              controller.canvas.clientWidth = 328;
              controller.canvas.clientHeight = 180;
              controller.canvas.textDraws = [];
              controller.draw();
              for (const call of controller.canvas.textDraws) {
                assert.ok(call.x >= 0, call.text + ' starts outside the canvas');
                assert.ok(call.x + call.text.length * 13 <= 328,
                  call.text + ' exceeds the 328px canvas');
              }
            }
            const forceRows = force.canvas.textDraws.filter(call => call.text.startsWith('■'));
            assert.equal(forceRows.length, 4);
            for (let index = 1; index < forceRows.length; index += 1) {
              assert.ok(forceRows[index].y - forceRows[index - 1].y >= 13);
            }
            const energyRows = energy.canvas.textDraws.filter(call =>
              call.text === '动能 K' || call.text === '势能 U' || call.text === '总能量 E');
            assert.deepEqual(energyRows.map(call => call.text), ['动能 K', '势能 U', '总能量 E']);
            for (let index = 1; index < energyRows.length; index += 1) {
              assert.ok(energyRows[index].y - energyRows[index - 1].y >= 13);
            }
            console.log(JSON.stringify({width: 328, forceRows: forceRows.length, energyRows: energyRows.length}));
        """)
        self.assertEqual(data, {"width": 328, "forceRows": 4, "energyRows": 3})

    def test_chapter_three_demos_keep_independent_returned_controllers(self):
        data = self.run_demos([
            REPO_DIR / "book/assets/demos/physics05.js",
            REPO_DIR / "book/assets/demos/physics06.js",
        ], """
            const assert = require('node:assert/strict');
            const forceRoot = document.getElementById('force-balance');
            const energyRoot = document.getElementById('energy-ledger');
            const force = mounts[0];
            const energy = mounts[1];
            assert.equal(mounts.length, 2);
            assert.notEqual(force.canvas, energy.canvas);
            force.step();
            assert.equal(force.getState().frame, 1);
            assert.equal(energy.getState().frame, 0);
            window.RTPhysicsUI.step();
            assert.equal(force.getState().frame, 1);
            assert.equal(energy.getState().frame, 1);
            force.reset();
            assert.equal(force.getState().frame, 0);
            assert.equal(energy.getState().frame, 1);
            energy.reset();
            assert.equal(energy.getState().frame, 0);
            assert.match(readout(forceRoot).textContent, /终端速度/);
            assert.match(readout(energyRoot).textContent, /相对能量损失/);
            console.log(JSON.stringify({independent: true, latestMount: true}));
        """)
        self.assertEqual(data, {"independent": True, "latestMount": True})

    def test_spring_oscillator_uses_equilibrium_coordinates_and_shared_core(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics07.js"], """
            const assert = require('node:assert/strict');
            assert.equal(mounts.length, 1);
            const root = document.getElementById('spring-oscillator');
            const toolbar = root.children.find(element => element.className === 'demo-toolbar physics-controls');
            const oscillator = mounts[0];
            assert.equal(toolbar.children[1].textContent, '运行');
            assert.equal(pending.size, 0);
            assert.deepEqual(inputs(root).map(input => Number(input.value)), [1, 20, 0, 0.5]);
            let state = oscillator.getState();
            assert.equal(state.x, 0.5);
            assert.equal(state.v, 0);
            assert.ok(Math.abs(state.staticExtension - 0.4905) < 1e-12);
            assert.equal(state.totalEnergy, 2.5);
            assert.ok(Math.abs(state.periodApprox - 2 * Math.PI * Math.sqrt(1 / 20)) < 1e-12);
            assert.ok(Math.abs(state.stepHint - 2 / Math.sqrt(20)) < 1e-12);
            assert.equal(state.substeps, 4);
            assert.equal(state.substepDt, 1 / 240);
            assert.match(oscillator.canvas.attributes['aria-label'], /竖直.*静平衡.*力/);
            springCalls.length = 0;
            oscillator.step();
            state = oscillator.getState();
            assert.equal(springCalls.length, 5);
            assert.deepEqual(springCalls[0], [0.5, 0, 20, 0]);
            assert.equal(state.frame, 1);
            assert.ok(Math.abs(state.time - 1 / 60) < 1e-12);
            assert.ok(Math.abs(state.v - (-0.16652202087981438)) < 1e-12);
            assert.ok(Math.abs(state.x - 0.4982647929669153) < 1e-12);
            assert.ok(Math.abs(state.force - (-9.965295859338305)) < 1e-12);
            assert.ok(Math.abs(state.totalEnergy - 2.496542830822578) < 1e-12);
            const text = readout(root).textContent;
            for (const marker of ['小振幅无阻尼周期近似', '当前伸长', '速度', '总能量', '稳定步长上限']) {
                assert.ok(text.includes(marker), marker);
            }
            assert.doesNotMatch(text, /NaN|Infinity/);
            console.log(JSON.stringify({x: state.x, v: state.v, energy: state.totalEnergy}));
        """, setup="""
            const sharedSpringForce = window.RTPhysics.springForce;
            const springCalls = [];
            window.RTPhysics.springForce = (...args) => {
                springCalls.push(args);
                return sharedSpringForce(...args);
            };
        """)
        self.assertAlmostEqual(data["x"], 0.4982647929669153)
        self.assertAlmostEqual(data["v"], -0.16652202087981438)
        self.assertAlmostEqual(data["energy"], 2.496542830822578)

    def test_spring_oscillator_controls_reset_and_extremes_stay_finite_for_720_steps(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics07.js"], """
            const assert = require('node:assert/strict');
            const oscillator = mounts[0];
            const root = document.getElementById('spring-oscillator');
            const controls = inputs(root);
            const cases = [[1, 20, 0, 0.5], [0.5, 80, 0, 1.5], [0.5, 80, 8, 1.5],
                            [2, 1, 0, 1.5], [2, 1, 8, 1.5]];
            for (const values of cases) {
                values.forEach((value, index) => { controls[index].value = String(value); });
                controls[0].dispatch('input');
                let state = oscillator.getState();
                assert.equal(state.mass, values[0]);
                assert.equal(state.k, values[1]);
                assert.equal(state.damping, values[2]);
                assert.equal(state.x, values[3]);
                assert.equal(state.v, 0);
                assert.equal(state.frame, 0);
                for (let frame = 0; frame < 720; frame += 1) {
                    oscillator.step();
                    state = oscillator.getState();
                    for (const key of ['x', 'v', 'force', 'kinetic', 'potential', 'totalEnergy',
                                       'periodApprox', 'staticExtension', 'stepHint', 'time']) {
                        assert.ok(Number.isFinite(state[key]), key + ' at frame ' + frame);
                    }
                }
                assert.equal(state.frame, 720);
                assert.equal(state.time, 12);
                assert.equal(state.finished, true);
                assert.equal(pending.size, 0);
                const toolbar = root.children.find(element =>
                    element.className === 'demo-toolbar physics-controls');
                assert.equal(toolbar.children[1].textContent, '运行');
                oscillator.setRunning(true);
                tick(0);
                tick(1000 / 60);
                assert.equal(pending.size, 0, 'restarting a completed demo must not keep a no-op RAF');
                assert.equal(toolbar.children[1].textContent, '运行');
                assert.equal(oscillator.getState().frame, 720);
                oscillator.setRunning(false);
                assert.doesNotMatch(readout(root).textContent, /NaN|Infinity/);
                oscillator.canvas.pathDraws = [];
                oscillator.canvas.textDraws = [];
            }
            oscillator.reset();
            assert.deepEqual(controls.map(input => Number(input.value)), [1, 20, 0, 0.5]);
            assert.equal(oscillator.getState().frame, 0);
            oscillator.setRunning(true);
            assert.equal(pending.size, 1, 'reset must allow playback to start normally');
            const toolbar = root.children.find(element =>
                element.className === 'demo-toolbar physics-controls');
            assert.equal(toolbar.children[1].textContent, '暂停');
            oscillator.setRunning(false);
            assert.equal(pending.size, 0);
            oscillator.step();
            const first = JSON.parse(JSON.stringify(oscillator.getState()));
            oscillator.reset();
            oscillator.step();
            assert.deepEqual(JSON.parse(JSON.stringify(oscillator.getState())), first);
            console.log(JSON.stringify({finite: true, deterministic: true, restartStopped: true,
                resetPlayback: true, cases: cases.length}));
        """)
        self.assertEqual(data, {"finite": True, "deterministic": True, "restartStopped": True,
                                "resetPlayback": True, "cases": 5})

    def test_spring_cloth_has_six_fixed_particles_and_structural_springs_only(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics08.js"], """
            const assert = require('node:assert/strict');
            assert.equal(mounts.length, 1);
            const root = document.getElementById('spring-cloth');
            const toolbar = root.children.find(element => element.className === 'demo-toolbar physics-controls');
            const cloth = mounts[0];
            assert.equal(toolbar.children[1].textContent, '运行');
            assert.equal(pending.size, 0);
            assert.deepEqual(inputs(root).map(input => Number(input.value)), [1.2, 60]);
            let state = cloth.getState();
            assert.equal(state.particles.length, 24);
            assert.equal(state.springs.length, 38);
            assert.ok(state.springs.every(spring => spring.kind === 'structural'));
            assert.equal(state.fixedCount, 6);
            assert.equal(JSON.stringify(state.particles.filter(p => p.fixed).map(p => p.row)),
                JSON.stringify([0, 0, 0, 0, 0, 0]));
            assert.equal(state.particles.filter(p => !p.fixed).length, 18);
            assert.equal(state.maxStretch, 0);
            assert.equal(state.maxSpeed, 0);
            assert.equal(state.finite, true);
            assert.equal(state.substeps, 4);
            assert.equal(state.substepDt, 1 / 240);
            assert.equal(state.springs.filter(spring => spring.restLength === 0.35).length, 20);
            assert.equal(state.springs.filter(spring => spring.restLength === 0.3).length, 18);
            const anchors = state.particles.filter(p => p.fixed).map(p => [p.x, p.y]);
            assert.match(cloth.canvas.attributes['aria-label'], /结构弹簧.*六点固定.*当前伸长.*弹力方向/);
            assert.doesNotMatch(cloth.canvas.attributes['aria-label'], /风力/);
            cloth.step();
            state = cloth.getState();
            assert.equal(springCalls, 38 * 4);
            assert.equal(state.frame, 1);
            assert.ok(state.maxStretch > 0);
            assert.ok(state.maxSpeed > 0);
            assert.ok(state.particles[23].vx > 0);
            assert.equal(JSON.stringify(state.particles.filter(p => p.fixed).map(p => [p.x, p.y])),
                JSON.stringify(anchors));
            assert.ok(state.particles.filter(p => p.fixed).every(p => p.vx === 0 && p.vy === 0));
            const text = readout(root).textContent;
            for (const marker of ['最大伸长', '最大速度', '固定点数', '有限值状态']) assert.ok(text.includes(marker));
            assert.doesNotMatch(text, /NaN|Infinity/);
            console.log(JSON.stringify({fixed: state.fixedCount, springs: state.springs.length,
                stretch: state.maxStretch, speed: state.maxSpeed}));
        """, setup="""
            const sharedSpringForce = window.RTPhysics.springForce;
            let springCalls = 0;
            window.RTPhysics.springForce = (...args) => {
                springCalls += 1;
                return sharedSpringForce(...args);
            };
        """)
        self.assertEqual(data["fixed"], 6)
        self.assertEqual(data["springs"], 38)
        self.assertGreater(data["stretch"], 0)
        self.assertGreater(data["speed"], 0)

    def test_spring_cloth_controls_and_extremes_stay_finite_for_720_steps(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics08.js"], """
            const assert = require('node:assert/strict');
            const cloth = mounts[0];
            const root = document.getElementById('spring-cloth');
            const controls = inputs(root);
            const cases = [[1.2, 60], [0, 120], [2, 20]];
            for (const values of cases) {
                values.forEach((value, index) => { controls[index].value = String(value); });
                controls[0].dispatch('input');
                let state = cloth.getState();
                assert.equal(state.wind, values[0]);
                assert.equal(state.k, values[1]);
                assert.equal(state.frame, 0);
                for (let frame = 0; frame < 720; frame += 1) {
                    cloth.step();
                    state = cloth.getState();
                    assert.equal(state.finite, true, 'finite at frame ' + frame);
                    assert.equal(state.fixedCount, 6);
                    for (const particle of state.particles) {
                        for (const key of ['x', 'y', 'vx', 'vy']) {
                            assert.ok(Number.isFinite(particle[key]), key + ' at frame ' + frame);
                        }
                    }
                    for (const key of ['maxStretch', 'maxSpeed', 'time']) {
                        assert.ok(Number.isFinite(state[key]), key + ' at frame ' + frame);
                    }
                }
                assert.equal(state.frame, 720);
                assert.equal(state.time, 12);
                assert.equal(state.finished, true);
                assert.equal(pending.size, 0);
                const toolbar = root.children.find(element =>
                    element.className === 'demo-toolbar physics-controls');
                assert.equal(toolbar.children[1].textContent, '运行');
                cloth.setRunning(true);
                tick(0);
                tick(1000 / 60);
                assert.equal(pending.size, 0, 'restarting a completed demo must not keep a no-op RAF');
                assert.equal(toolbar.children[1].textContent, '运行');
                assert.equal(cloth.getState().frame, 720);
                cloth.setRunning(false);
                assert.doesNotMatch(readout(root).textContent, /NaN|Infinity/);
                cloth.canvas.pathDraws = [];
                cloth.canvas.textDraws = [];
            }
            cloth.reset();
            assert.deepEqual(controls.map(input => Number(input.value)), [1.2, 60]);
            assert.equal(cloth.getState().frame, 0);
            cloth.setRunning(true);
            assert.equal(pending.size, 1, 'reset must allow playback to start normally');
            const toolbar = root.children.find(element =>
                element.className === 'demo-toolbar physics-controls');
            assert.equal(toolbar.children[1].textContent, '暂停');
            cloth.setRunning(false);
            assert.equal(pending.size, 0);
            cloth.step();
            const first = JSON.parse(JSON.stringify(cloth.getState()));
            cloth.reset();
            cloth.step();
            assert.deepEqual(JSON.parse(JSON.stringify(cloth.getState())), first);
            console.log(JSON.stringify({finite: true, deterministic: true, restartStopped: true,
                resetPlayback: true, cases: cases.length}));
        """)
        self.assertEqual(data, {"finite": True, "deterministic": True, "restartStopped": True,
                                "resetPlayback": True, "cases": 3})

    def test_spring_cloth_applies_equal_and_opposite_forces_to_a_free_horizontal_pair(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics08.js"], """
            const assert = require('node:assert/strict');
            const cloth = mounts[0];
            const cases = [{name: 'extension', extension: 0.05, force: -3.5},
                           {name: 'compression', extension: -0.05, force: 2.25}];
            for (const fixture of cases) {
                cloth.reset();
                const state = cloth.getState();
                const selected = state.springs.find(spring => spring.a === 7 && spring.b === 8);
                assert.ok(selected);
                assert.equal(selected.restLength, 0.35);
                const a = state.particles[selected.a];
                const b = state.particles[selected.b];
                assert.equal(a.fixed, false);
                assert.equal(b.fixed, false);
                state.wind = 0;
                state.damping = 0;
                a.x = 1;
                a.y = 1;
                b.x = 1 + selected.restLength + fixture.extension;
                b.y = 1;
                selectedSpringIndex = state.springs.indexOf(selected);
                knownForce = fixture.force;
                selectedCalls = 0;
                totalCalls = 0;
                cloth.setRunning(true);
                assert.equal(pending.size, 1);
                tick(0);
                tick(1000 / 60);
                assert.equal(state.frame, 1, 'one transport update must advance one fixed frame');
                cloth.setRunning(false);
                assert.equal(pending.size, 0);
                assert.equal(totalCalls, 38 * 4);
                assert.equal(selectedCalls, 4, 'selected spring must be integrated in all four substeps');
                assert.equal(selected.force, knownForce);

                const gravityForce = 9.81 * state.particleMass;
                const close = (actual, expected) =>
                    assert.ok(Math.abs(actual - expected) <= 1e-12, actual + ' != ' + expected);
                close(a.fx, -knownForce);
                close(b.fx, knownForce);
                close(a.fx + b.fx, 0);
                close(a.fy - gravityForce, 0);
                close(b.fy - gravityForce, 0);
                close((a.fy - gravityForce) + (b.fy - gravityForce), 0);

                const elasticVelocity = knownForce * 4 * state.substepDt / state.particleMass;
                const gravityVelocity = 9.81 * 4 * state.substepDt;
                close(a.vx, -elasticVelocity);
                close(b.vx, elasticVelocity);
                close(a.vx + b.vx, 0);
                close(a.vy, gravityVelocity);
                close(b.vy, gravityVelocity);
                close(a.vy - gravityVelocity, 0);
                close(b.vy - gravityVelocity, 0);
            }
            console.log(JSON.stringify({cases: cases.map(fixture => fixture.name),
                substeps: 4, equalAndOpposite: true}));
        """, setup="""
            const sharedSpringForce = window.RTPhysics.springForce;
            let selectedCalls = 0;
            let totalCalls = 0;
            let selectedSpringIndex = -1;
            let knownForce = 0;
            window.RTPhysics.springForce = (...args) => {
                totalCalls += 1;
                if ((totalCalls - 1) % 38 === selectedSpringIndex) {
                    selectedCalls += 1;
                    return knownForce;
                }
                return 0;
            };
        """)
        self.assertEqual(data, {"cases": ["extension", "compression"], "substeps": 4,
                                "equalAndOpposite": True})

    def test_chapter_four_demo_geometry_and_labels_fit_narrow_canvas(self):
        data = self.run_demos([
            REPO_DIR / "book/assets/demos/physics07.js",
            REPO_DIR / "book/assets/demos/physics08.js",
        ], """
            const assert = require('node:assert/strict');
            const oscillator = mounts[0];
            const cloth = mounts[1];
            for (const controller of [oscillator, cloth]) {
                controller.canvas.clientWidth = 328;
                controller.canvas.clientHeight = 180;
            }
            const oscillatorCases = [[1, 20, 0, 0.5], [0.5, 80, 8, 1.5],
                                     [2, 1, 0, 1.5], [2, 1, 8, 1.5]];
            const oscillatorControls = inputs(document.getElementById('spring-oscillator'));
            for (const values of oscillatorCases) {
                values.forEach((value, index) => { oscillatorControls[index].value = String(value); });
                oscillatorControls[0].dispatch('input');
                oscillator.canvas.pathDraws = [];
                oscillator.canvas.textDraws = [];
                oscillator.draw();
                for (const call of oscillator.canvas.pathDraws) {
                    assert.ok(Number.isFinite(call.x) && Number.isFinite(call.y));
                    assert.ok(call.x >= 0 && call.x <= 328, 'oscillator path x outside canvas: ' + call.x);
                    assert.ok(call.y >= 0 && call.y <= 180, 'oscillator path y outside canvas: ' + call.y);
                }
                for (const call of oscillator.canvas.textDraws) {
                    assert.ok(call.x >= 0, call.text + ' starts outside the canvas');
                    assert.ok(call.x + call.text.length * 13 <= 328,
                        call.text + ' exceeds the 328px canvas');
                    assert.ok(call.y >= 0 && call.y <= 180, call.text + ' exceeds canvas height');
                }
                if (values[0] === 1) {
                    const spring = oscillator.canvas.pathDraws.slice(0, 9);
                    assert.equal(spring.length, 9);
                    assert.equal(spring[0].op, 'moveTo');
                    assert.equal(spring[0].x, spring[8].x, 'vertical oscillator spring must stay on one axis');
                    assert.ok(spring[8].y > spring[0].y);
                    assert.ok(spring.every(call => Math.abs(call.x - spring[0].x) <= 6));
                    assert.ok(oscillator.canvas.textDraws.some(call => call.text.includes('静平衡')));
                    assert.ok(oscillator.canvas.textDraws.some(call => call.text.includes('F=')));
                }
            }

            const clothCases = [[1.2, 60], [0, 120], [2, 20]];
            const clothControls = inputs(document.getElementById('spring-cloth'));
            for (const values of clothCases) {
                values.forEach((value, index) => { clothControls[index].value = String(value); });
                clothControls[0].dispatch('input');
                cloth.canvas.pathDraws = [];
                cloth.canvas.textDraws = [];
                cloth.draw();
                for (const call of cloth.canvas.pathDraws) {
                    assert.ok(Number.isFinite(call.x) && Number.isFinite(call.y));
                    assert.ok(call.x >= 0 && call.x <= 328, 'cloth path x outside canvas: ' + call.x);
                    assert.ok(call.y >= 0 && call.y <= 180, 'cloth path y outside canvas: ' + call.y);
                }
                for (const call of cloth.canvas.textDraws) {
                    assert.ok(call.x >= 0, call.text + ' starts outside the canvas');
                    assert.ok(call.x + call.text.length * 13 <= 328,
                        call.text + ' exceeds the 328px canvas');
                    assert.ok(call.y >= 0 && call.y <= 180, call.text + ' exceeds canvas height');
                }
                const structuralPaths = cloth.canvas.pathDraws.slice(0, 76);
                assert.equal(structuralPaths.length, 76);
                const arcs = cloth.canvas.pathDraws.filter(call => call.op === 'arc');
                assert.equal(arcs.length, 24);
                const atArc = (call, arc) => Math.hypot(call.x - arc.x, call.y - arc.y) <= 1e-9;
                const connected = new Set();
                for (let index = 0; index < structuralPaths.length; index += 2) {
                    const start = structuralPaths[index];
                    const end = structuralPaths[index + 1];
                    assert.equal(start.op, 'moveTo');
                    assert.equal(end.op, 'lineTo');
                    const a = arcs.findIndex(arc => atArc(start, arc));
                    const b = arcs.findIndex(arc => atArc(end, arc));
                    assert.ok(a >= 0 && b >= 0, 'structural segment must join recorded particle centers');
                    assert.notEqual(a, b, 'structural segment must connect two different particles');
                    connected.add(a);
                    connected.add(b);
                }
                assert.equal(connected.size, 24, 'every particle center must participate in the mesh');
                for (const marker of ['结构弹簧', '剪切/弯曲未启用', '固定点', '静止长度']) {
                    assert.ok(cloth.canvas.textDraws.some(call => call.text.includes(marker)), marker);
                }
            }
            console.log(JSON.stringify({width: 328, structuralSprings: 38, particles: 24,
                oscillatorCases: oscillatorCases.length, clothCases: clothCases.length}));
        """)
        self.assertEqual(data, {"width": 328, "structuralSprings": 38, "particles": 24,
                                "oscillatorCases": 4, "clothCases": 3})

    def test_chapter_four_demos_keep_independent_returned_controllers(self):
        data = self.run_demos([
            REPO_DIR / "book/assets/demos/physics07.js",
            REPO_DIR / "book/assets/demos/physics08.js",
        ], """
            const assert = require('node:assert/strict');
            const oscillatorRoot = document.getElementById('spring-oscillator');
            const clothRoot = document.getElementById('spring-cloth');
            const oscillator = mounts[0];
            const cloth = mounts[1];
            assert.equal(mounts.length, 2);
            assert.notEqual(oscillator.canvas, cloth.canvas);
            for (let frame = 0; frame < 720; frame += 1) oscillator.step();
            assert.equal(oscillator.getState().finished, true);
            oscillator.setRunning(true);
            cloth.setRunning(true);
            assert.equal(pending.size, 2);
            tick(0);
            tick(0);
            tick(1000 / 60);
            assert.equal(pending.size, 1, 'completed oscillator must leave the cloth RAF running');
            assert.equal(oscillator.getState().frame, 720);
            assert.equal(cloth.getState().frame, 0);
            assert.equal(oscillatorRoot.children.find(element =>
                element.className === 'demo-toolbar physics-controls').children[1].textContent, '运行');
            assert.equal(clothRoot.children.find(element =>
                element.className === 'demo-toolbar physics-controls').children[1].textContent, '暂停');
            oscillator.reset();
            assert.equal(oscillator.getState().frame, 0);
            assert.equal(cloth.getState().frame, 0);
            assert.equal(pending.size, 1);
            tick(1000 / 60);
            assert.equal(cloth.getState().frame, 1);
            oscillator.step();
            assert.equal(oscillator.getState().frame, 1);
            assert.equal(cloth.getState().frame, 1);
            cloth.reset();
            assert.equal(cloth.getState().frame, 0);
            assert.equal(oscillator.getState().frame, 1);
            oscillator.reset();
            assert.equal(oscillator.getState().frame, 0);
            assert.equal(pending.size, 0);
            assert.match(readout(oscillatorRoot).textContent, /周期近似/);
            assert.match(readout(clothRoot).textContent, /固定点数/);
            console.log(JSON.stringify({independent: true}));
        """)
        self.assertEqual(data, {"independent": True})

    def test_pbd_rope_projects_eight_particles_with_fixed_ends_and_selected_dt(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics09.js"], """
            const assert = require('node:assert/strict');
            const worked = window.RTPhysics.projectDistance(
                {x: 0, y: 0, vx: 0, vy: 0, w: 1},
                {x: 3, y: 0, vx: 0, vy: 0, w: 1}, 2, 1);
            assert.deepEqual(JSON.parse(JSON.stringify(worked.a)), {x: 0.5, y: 0, vx: 0, vy: 0, w: 1});
            assert.deepEqual(JSON.parse(JSON.stringify(worked.b)), {x: 2.5, y: 0, vx: 0, vy: 0, w: 1});
            const workedCalls = projectionCalls;
            assert.equal(mounts.length, 1);
            const root = document.getElementById('pbd-rope');
            const rope = mounts[0];
            const toolbar = root.children.find(element => element.className === 'demo-toolbar physics-controls');
            assert.equal(toolbar.children[1].textContent, '运行');
            assert.equal(pending.size, 0);
            assert.deepEqual(inputs(root).map(input => Number(input.value)), [8, 1, 1 / 120]);
            let state = rope.getState();
            assert.equal(state.particles.length, 8);
            assert.equal(state.constraints.length, 7);
            assert.equal(state.particles.filter(p => p.w === 0).length, 2);
            const initialEnds = state.particles.filter(p => p.w === 0).map(p => [p.x, p.y]);
            assert.equal(initialEnds[0][0], 0);
            assert.ok(Math.abs(initialEnds[1][0] - 2.45) < 1e-12);
            assert.ok(initialEnds.every(point => point[1] === 1.5));
            assert.equal(state.fixedCount, 2);
            assert.equal(state.maxError, 0);
            assert.match(rope.canvas.attributes['aria-label'], /绳子.*固定.*拖拽/);
            rope.step();
            state = rope.getState();
            assert.equal(projectionCalls - workedCalls, 7 * 8);
            assert.equal(state.frame, 1);
            assert.equal(state.dt, 1 / 120);
            assert.equal(state.time, 1 / 120);
            assert.ok(state.errorBefore > 0);
            assert.ok(state.errorAfter >= 0 && state.errorAfter < state.errorBefore);
            assert.equal(state.maxError, state.errorAfter);
            assert.ok(state.particles.slice(1, -1).every(p => p.vy > 0));
            const ends = state.particles.filter(p => p.w === 0);
            assert.equal(JSON.stringify(ends.map(p => [p.x, p.y, p.vx, p.vy])),
                JSON.stringify(initialEnds.map(point => [point[0], point[1], 0, 0])));
            const text = readout(root).textContent;
            for (const marker of ['最大相对约束误差', '迭代前', '迭代后', '固定端数量', '物理时间步']) {
                assert.ok(text.includes(marker), marker);
            }
            assert.doesNotMatch(text, /NaN|Infinity/);
            console.log(JSON.stringify({calls: projectionCalls - workedCalls, before: state.errorBefore,
                after: state.errorAfter, time: state.time}));
        """, setup="""
            const sharedProject = window.RTPhysics.projectDistance;
            let projectionCalls = 0;
            window.RTPhysics.projectDistance = (...args) => {
                projectionCalls += 1;
                return sharedProject(...args);
            };
        """)
        self.assertEqual(data["calls"], 56)
        self.assertGreater(data["before"], data["after"])
        self.assertAlmostEqual(data["time"], 1 / 120)

    def test_pbd_rope_drag_clamps_bounds_and_reconstruction_has_no_artificial_velocity(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics09.js"], """
            const assert = require('node:assert/strict');
            const rope = mounts[0];
            const root = document.getElementById('pbd-rope');
            const controls = inputs(root);
            controls[1].value = '0'; controls[1].dispatch('input');
            assert.equal(rope.getState().stiffness, 0);
            rope.canvas.clientWidth = 328;
            rope.canvas.clientHeight = 180;
            rope.draw();
            const middle = rope.canvas.pathDraws.filter(call => call.op === 'arc')[3];
            rope.canvas.dispatch('pointerdown', {clientX: middle.x, clientY: middle.y, pointerId: 9});
            rope.canvas.dispatch('pointermove', {clientX: middle.x + 20, clientY: middle.y + 30, pointerId: 9});
            rope.canvas.dispatch('pointerup', {pointerId: 9});
            let dragged = rope.getState().particles[3];
            assert.equal(dragged.vx, 0);
            assert.equal(dragged.vy, 0);
            assert.equal(rope.getState().dragging, null);
            const reference = {x: dragged.x, y: dragged.y};
            rope.step();
            dragged = rope.getState().particles[3];
            assert.equal(dragged.x, reference.x);
            assert.ok(Math.abs(dragged.y - (reference.y + 9.81 / 120 * (1 / 120))) < 1e-12);
            assert.ok(Math.abs(dragged.vy - 9.81 / 120) < 1e-12,
                'dragging must not become a large reconstructed velocity');

            rope.reset();
            rope.canvas.dispatch('pointerdown', {clientX: middle.x, clientY: middle.y, pointerId: 10});
            rope.canvas.dispatch('pointermove', {clientX: -1000, clientY: 1000, pointerId: 10});
            rope.canvas.dispatch('pointerup', {pointerId: 10});
            dragged = rope.getState().particles[3];
            assert.ok(dragged.x >= 0 && dragged.x <= 2.45);
            assert.ok(dragged.y >= 0.95 && dragged.y <= 2);
            assert.equal(rope.getState().particles[0].x, 0);
            assert.ok(Math.abs(rope.getState().particles[7].x - 2.45) < 1e-12);
            rope.reset();
            const resetParticle = rope.getState().particles[3];
            assert.ok(Math.abs(resetParticle.x - 1.05) < 1e-12);
            assert.equal(resetParticle.y, 1.5);
            assert.equal(resetParticle.vx, 0);
            assert.equal(resetParticle.vy, 0);
            assert.equal(resetParticle.w, 1);
            assert.deepEqual(controls.map(input => Number(input.value)), [8, 1, 1 / 120]);
            assert.doesNotMatch(readout(root).textContent, /NaN|Infinity/);
            console.log(JSON.stringify({dragReference: true, bounded: true, reset: true}));
        """)
        self.assertEqual(data, {"dragReference": True, "bounded": True, "reset": True})

    def test_pbd_rope_pauses_on_coincident_constraint_instead_of_faking_a_direction(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics09.js"], """
            const assert = require('node:assert/strict');
            const rope = mounts[0];
            const root = document.getElementById('pbd-rope');
            const state = rope.getState();
            state.particles[1].x = state.particles[2].x;
            state.particles[1].y = state.particles[2].y;
            rope.step();
            assert.equal(state.invalid, true);
            assert.equal(state.finite, false);
            assert.equal(state.finished, true);
            assert.equal(pending.size, 0);
            const warning = children(root).find(element => element.attributes.role === 'status');
            assert.equal(warning.hidden, false);
            assert.match(warning.textContent, /零长度|重合/);
            assert.match(warning.textContent, /重置/);
            const stopped = JSON.parse(JSON.stringify(state));
            rope.step();
            assert.deepEqual(JSON.parse(JSON.stringify(rope.getState())), stopped);
            rope.reset();
            assert.equal(rope.getState().invalid, false);
            assert.equal(rope.getState().finite, true);
            assert.equal(warning.hidden, true);
            console.log(JSON.stringify({paused: true, reset: true}));
        """)
        self.assertEqual(data, {"paused": True, "reset": True})

    def test_pbd_rope_zero_stiffness_pauses_at_observation_boundary_with_finite_drawing(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics09.js"], """
            const assert = require('node:assert/strict');
            const rope = mounts[0];
            const root = document.getElementById('pbd-rope');
            const controls = inputs(root);
            const toolbar = root.children.find(element =>
                element.className === 'demo-toolbar physics-controls');
            const warning = children(root).find(element => element.attributes.role === 'status');
            const results = [];
            for (const expected of [{dt: 1 / 240, frame: 77}, {dt: 1 / 60, frame: 19}]) {
                controls[1].value = '0';
                controls[2].value = String(expected.dt);
                controls[1].dispatch('input');
                assert.equal(rope.getState().stiffness, 0);
                assert.equal(rope.getState().dt, expected.dt);
                let callbacks = 0;
                let now = 0;
                rope.setRunning(true);
                assert.equal(pending.size, 1);
                while (pending.size > 0 && callbacks < 500) {
                    tick(now);
                    now += 1000 / 60;
                    callbacks += 1;
                }
                assert.equal(pending.size, 0, 'observation stop must leave no pending RAF');
                assert.ok(callbacks >= expected.frame + 1 && callbacks <= expected.frame + 2,
                    'pause on the first crossing physical step without a no-op RAF tail');
                assert.equal(toolbar.children[1].textContent, '运行');
                const state = rope.getState();
                assert.equal(state.finished, true);
                assert.equal(state.invalid, false, 'crossing a view boundary is not invalid physics');
                assert.equal(state.frame, expected.frame);
                assert.ok(Math.abs(state.time - expected.frame * expected.dt) < 1e-12);
                assert.ok(state.time < 8, 'zero stiffness must stop before the eight-second endpoint');
                assert.equal(state.finite, true);
                for (const key of ['iterations', 'stiffness', 'dt', 'gravity', 'errorBefore',
                    'errorAfter', 'maxError', 'time', 'frame']) {
                    assert.ok(Number.isFinite(state[key]), key);
                }
                const free = state.particles.filter(particle => particle.w > 0);
                assert.equal(free.length, 6);
                assert.ok(free.every(particle => particle.y > 2), 'retain the crossing step beyond maxY');
                const crossingY = 1.5 + 9.81 * expected.dt * expected.dt *
                    expected.frame * (expected.frame + 1) / 2;
                const previousY = 1.5 + 9.81 * expected.dt * expected.dt *
                    (expected.frame - 1) * expected.frame / 2;
                assert.ok(previousY <= 2 && crossingY > 2, 'hand-derived first crossing');
                assert.ok(Math.max(...free.map(particle => particle.y)) < 2 + 9.81 * expected.dt,
                    'only a small final-step overshoot is allowed');
                for (const particle of state.particles) {
                    for (const key of ['x', 'y', 'vx', 'vy', 'w']) {
                        assert.ok(Number.isFinite(particle[key]), key);
                    }
                }
                assert.equal(warning.hidden, false);
                assert.equal(warning.attributes['aria-live'], 'polite');
                assert.match(warning.textContent, /观察视图边界暂停/);
                assert.match(warning.textContent, /不是碰撞/);
                assert.match(warning.textContent, /重置/);
                assert.doesNotMatch(readout(root).textContent, /NaN|Infinity/);
                const metricValues = children(root).filter(element => element.className === 'metric')
                    .map(element => Number(element.children[1].textContent));
                assert.equal(metricValues.length, 11);
                assert.ok(metricValues.every(Number.isFinite));
                const warningText = warning.textContent;

                rope.canvas.clientWidth = 328;
                rope.canvas.clientHeight = 180;
                rope.canvas.pathDraws = [];
                rope.canvas.textDraws = [];
                rope.draw();
                assert.equal(rope.canvas.width, 328);
                assert.equal(rope.canvas.height, 180);
                for (const call of rope.canvas.pathDraws) {
                    assert.ok(Number.isFinite(call.x) && Number.isFinite(call.y));
                    assert.ok(call.x >= 0 && call.x <= 328, 'path x outside canvas: ' + call.x);
                    assert.ok(call.y >= 0 && call.y <= 180, 'path y outside canvas: ' + call.y);
                }
                for (const call of rope.canvas.textDraws) {
                    assert.ok(Number.isFinite(call.x) && Number.isFinite(call.y));
                    assert.ok(call.x >= 0 && call.x + call.text.length * 13 <= 328, call.text);
                    assert.ok(call.y >= 0 && call.y <= 180, call.text);
                }

                const stopped = JSON.parse(JSON.stringify(state));
                rope.step();
                assert.deepEqual(JSON.parse(JSON.stringify(rope.getState())), stopped);
                rope.setRunning(true);
                rope.setRunning(true);
                assert.equal(pending.size, 1);
                tick(now);
                tick(now + 1000 / 60);
                assert.equal(pending.size, 0, 'a finished play callback must stop itself');
                assert.deepEqual(JSON.parse(JSON.stringify(rope.getState())), stopped);

                rope.reset();
                assert.equal(warning.hidden, true);
                assert.equal(warning.textContent, '');
                assert.deepEqual(controls.map(input => Number(input.value)), [8, 1, 1 / 120]);
                assert.equal(rope.getState().finished, false);
                assert.equal(rope.getState().frame, 0);
                assert.equal(rope.getState().time, 0);
                rope.setRunning(true);
                assert.equal(pending.size, 1, 'reset must schedule normal playback');
                rope.setRunning(false);
                assert.equal(pending.size, 0);
                results.push({dt: expected.dt, frame: expected.frame, time: state.time,
                    callbacks, warning: warningText});
            }
            console.log(JSON.stringify(results));
        """)
        self.assertEqual([(item["dt"], item["frame"]) for item in data],
                         [(1 / 240, 77), (1 / 60, 19)])
        for item in data:
            self.assertGreaterEqual(item["callbacks"], item["frame"] + 1)
            self.assertLessEqual(item["callbacks"], item["frame"] + 2)
            self.assertLess(item["time"], 8)
            self.assertIn("不是碰撞", item["warning"])

    def test_pbd_rope_controls_finish_finitely_and_reset_deterministically(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics09.js"], """
            const assert = require('node:assert/strict');
            const rope = mounts[0];
            const root = document.getElementById('pbd-rope');
            const controls = inputs(root);
            const cases = [
                {values: [8, 1, 1 / 120], frames: 960, time: 8},
                {values: [1, 0, 1 / 240], frames: 77, time: 77 / 240},
                {values: [40, 0.25, 1 / 60], frames: 480, time: 8},
            ];
            for (const testCase of cases) {
                const values = testCase.values;
                values.forEach((value, index) => { controls[index].value = String(value); });
                controls[0].dispatch('input');
                let state = rope.getState();
                assert.equal(state.iterations, values[0]);
                assert.equal(state.stiffness, values[1]);
                assert.equal(state.dt, values[2]);
                assert.equal(state.frame, 0);
                assert.equal(state.time, 0);
                for (let frame = 0; frame < testCase.frames; frame += 1) {
                    rope.step();
                    state = rope.getState();
                    assert.equal(state.finite, true, 'finite at frame ' + frame);
                    assert.equal(state.fixedCount, 2);
                    for (const particle of state.particles) {
                        for (const key of ['x', 'y', 'vx', 'vy', 'w']) {
                            assert.ok(Number.isFinite(particle[key]), key + ' at frame ' + frame);
                        }
                    }
                    for (const key of ['maxError', 'errorBefore', 'errorAfter', 'time']) {
                        assert.ok(Number.isFinite(state[key]), key + ' at frame ' + frame);
                    }
                }
                assert.equal(state.frame, testCase.frames);
                assert.ok(Math.abs(state.time - testCase.time) < 1e-12);
                if (testCase.time === 8) {
                    assert.equal(state.time, 8, 'only full-duration cases reach the endpoint');
                }
                assert.equal(state.finished, true);
                assert.equal(state.invalid, false);
                assert.equal(pending.size, 0);
                const toolbar = root.children.find(element =>
                    element.className === 'demo-toolbar physics-controls');
                assert.equal(toolbar.children[1].textContent, '运行');
                rope.setRunning(true);
                tick(0);
                tick(1000 / 60);
                assert.equal(pending.size, 0, 'completed rope must not keep a no-op RAF');
                assert.equal(toolbar.children[1].textContent, '运行');
                assert.equal(rope.getState().frame, testCase.frames);
                rope.setRunning(false);
                rope.canvas.pathDraws = [];
                rope.canvas.textDraws = [];
            }
            rope.reset();
            assert.deepEqual(controls.map(input => Number(input.value)), [8, 1, 1 / 120]);
            assert.equal(rope.getState().finished, false);
            rope.step();
            const first = JSON.parse(JSON.stringify(rope.getState()));
            rope.reset();
            rope.step();
            assert.deepEqual(JSON.parse(JSON.stringify(rope.getState())), first);
            rope.setRunning(true);
            assert.equal(pending.size, 1, 'reset must allow transport playback');
            rope.setRunning(false);
            assert.equal(pending.size, 0);
            console.log(JSON.stringify({finite: true, deterministic: true, cases: cases.length}));
        """)
        self.assertEqual(data, {"finite": True, "deterministic": True, "cases": 3})

    def test_constraint_order_uses_identical_starts_real_projections_and_distinct_sweeps(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics10.js"], """
            const assert = require('node:assert/strict');
            assert.equal(mounts.length, 1);
            const order = mounts[0];
            const root = document.getElementById('constraint-order');
            const toolbar = root.children.find(element => element.className === 'demo-toolbar physics-controls');
            assert.equal(toolbar.children[1].textContent, '运行');
            assert.equal(pending.size, 0);
            assert.deepEqual(inputs(root).map(input => Number(input.value)), [0.25, 4]);
            let state = order.getState();
            assert.equal(JSON.stringify(state.runs.map(run => run.name)),
                JSON.stringify(['forward', 'reverse', 'alternating']));
            assert.equal(state.iteration, 0);
            assert.equal(state.time, 0);
            assert.ok(Math.abs(state.initialError - 0.285496013218244) < 1e-12);
            for (const run of state.runs) {
                assert.deepEqual(JSON.parse(JSON.stringify(run.particles)), [
                    {x: 0, y: 0, vx: 0, vy: 0, w: 0},
                    {x: 1.25, y: 0.3, vx: 0, vy: 0, w: 1},
                    {x: 2, y: 0, vx: 0, vy: 0, w: 0},
                ]);
            }
            order.step();
            state = order.getState();
            assert.equal(state.iteration, 1);
            assert.equal(state.time, 1 / 60);
            assert.equal(projectionCalls, 8);
            assert.ok(projectionCallsEveryReal);
            const expected = [
                [1.0248311883234593, 0.22146283826719806, 0.048486887516404176],
                [0.9448555004636517, 0.3274875314322394, 0.10479771820247374],
                [0.9774382498488091, 0.21122137139147892, 0.04414893602625658],
            ];
            state.runs.forEach((run, index) => {
                assert.ok(Math.abs(run.particles[1].x - expected[index][0]) < 1e-12,
                    run.name + ' x ' + run.particles[1].x + ' != ' + expected[index][0]);
                assert.ok(Math.abs(run.particles[1].y - expected[index][1]) < 1e-12,
                    run.name + ' y ' + run.particles[1].y + ' != ' + expected[index][1]);
                assert.ok(Math.abs(run.error - expected[index][2]) < 1e-12,
                    run.name + ' error ' + run.error + ' != ' + expected[index][2]);
                assert.equal(run.particles[0].w, 0);
                assert.equal(run.particles[2].w, 0);
                assert.deepEqual([run.particles[0].x, run.particles[0].y], [0, 0]);
                assert.deepEqual([run.particles[2].x, run.particles[2].y], [2, 0]);
            });
            assert.notEqual(state.runs[0].particles[1].x, state.runs[1].particles[1].x);
            const text = readout(root).textContent;
            for (const marker of ['初始最大相对约束误差', '正序最大相对约束误差', '逆序最大相对约束误差',
                                  '交替最大相对约束误差', '正序中点 x', '正序中点 y', '逆序中点 x',
                                  '逆序中点 y', '交替中点 x', '交替中点 y']) {
                assert.ok(text.includes(marker), marker);
            }
            assert.doesNotMatch(text, /NaN|Infinity/);
            console.log(JSON.stringify({calls: projectionCalls, positions: state.runs.map(run => run.particles[1]),
                errors: state.runs.map(run => run.error)}));
        """, setup="""
            const sharedProject = window.RTPhysics.projectDistance;
            let projectionCalls = 0;
            let projectionCallsEveryReal = true;
            window.RTPhysics.projectDistance = (...args) => {
                projectionCalls += 1;
                if (args[2] !== 1 || args[3] !== 1 || args[0].w + args[1].w !== 1) {
                    projectionCallsEveryReal = false;
                }
                return sharedProject(...args);
            };
        """)
        self.assertEqual(data["calls"], 8)
        self.assertNotEqual(data["positions"][0]["x"], data["positions"][1]["x"])

    def test_constraint_order_controls_converge_one_sweep_per_step_and_reset_deterministically(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics10.js"], """
            const assert = require('node:assert/strict');
            const order = mounts[0];
            const root = document.getElementById('constraint-order');
            const controls = inputs(root);
            const cases = [[0.25, 4], [0.1, 1], [0.8, 80]];
            for (const values of cases) {
                controls[0].value = String(values[0]); controls[1].value = String(values[1]);
                controls[0].dispatch('input');
                let state = order.getState();
                assert.equal(state.offset, values[0]);
                assert.equal(state.iterations, values[1]);
                assert.equal(state.iteration, 0);
                assert.ok(state.runs.every(run => run.particles[1].x === 1 + values[0]));
                for (let iteration = 1; iteration <= values[1]; iteration += 1) {
                    order.step();
                    state = order.getState();
                    assert.equal(state.iteration, iteration);
                    assert.equal(state.finite, true);
                    for (const run of state.runs) {
                        assert.ok(Number.isFinite(run.error) && run.error >= 0);
                        for (const particle of run.particles) {
                            for (const key of ['x', 'y', 'vx', 'vy', 'w']) assert.ok(Number.isFinite(particle[key]));
                        }
                    }
                }
                assert.ok(state.runs.every(run => run.error < state.initialError));
                assert.equal(state.finished, true);
                const finished = JSON.parse(JSON.stringify(state));
                order.step();
                assert.deepEqual(JSON.parse(JSON.stringify(order.getState())), finished);
                order.setRunning(true);
                tick(0); tick(1000 / 60);
                assert.equal(pending.size, 0);
                assert.equal(order.getState().iteration, values[1]);
                order.setRunning(false);
                assert.doesNotMatch(readout(root).textContent, /NaN|Infinity/);
            }
            order.reset();
            assert.deepEqual(controls.map(input => Number(input.value)), [0.25, 4]);
            assert.equal(order.getState().iteration, 0);
            assert.equal(order.getState().finished, false);
            order.step();
            const first = JSON.parse(JSON.stringify(order.getState()));
            order.reset(); order.step();
            assert.deepEqual(JSON.parse(JSON.stringify(order.getState())), first);
            order.reset();
            order.setRunning(true);
            tick(0); tick(1000 / 60);
            assert.equal(order.getState().iteration, 1);
            order.setRunning(false);
            console.log(JSON.stringify({finite: true, deterministic: true, cases: cases.length}));
        """)
        self.assertEqual(data, {"finite": True, "deterministic": True, "cases": 3})

    def test_chapter_five_demo_geometry_fits_narrow_canvas(self):
        data = self.run_demos([
            REPO_DIR / "book/assets/demos/physics09.js",
            REPO_DIR / "book/assets/demos/physics10.js",
        ], """
            const assert = require('node:assert/strict');
            const rope = mounts[0];
            const order = mounts[1];
            for (const controller of [rope, order]) {
                controller.canvas.clientWidth = 328;
                controller.canvas.clientHeight = 180;
            }
            for (let frame = 0; frame < 4; frame += 1) rope.step();
            for (let iteration = 0; iteration < 4; iteration += 1) order.step();
            for (const controller of [rope, order]) {
                controller.canvas.pathDraws = [];
                controller.canvas.textDraws = [];
                controller.draw();
                for (const call of controller.canvas.pathDraws) {
                    assert.ok(Number.isFinite(call.x) && Number.isFinite(call.y));
                    assert.ok(call.x >= 0 && call.x <= 328, 'path x outside canvas: ' + call.x);
                    assert.ok(call.y >= 0 && call.y <= 180, 'path y outside canvas: ' + call.y);
                }
                for (const call of controller.canvas.textDraws) {
                    assert.ok(call.x >= 0, call.text + ' starts outside canvas');
                    assert.ok(call.x + call.text.length * 13 <= 328, call.text + ' exceeds 328px canvas');
                    assert.ok(call.y >= 0 && call.y <= 180, call.text + ' exceeds canvas height');
                }
            }
            assert.equal(rope.canvas.pathDraws.filter(call => call.op === 'arc').length, 8);
            assert.equal(order.canvas.pathDraws.filter(call => call.op === 'arc').length, 9);
            assert.ok(rope.canvas.textDraws.some(call => call.text.includes('拖拽')));
            assert.ok(order.canvas.textDraws.some(call => call.text.includes('正序')));
            assert.ok(order.canvas.textDraws.some(call => call.text.includes('逆序')));
            assert.ok(order.canvas.textDraws.some(call => call.text.includes('交替')));
            console.log(JSON.stringify({width: 328, height: 180, ropeParticles: 8, orderParticles: 9}));
        """)
        self.assertEqual(data, {"width": 328, "height": 180, "ropeParticles": 8, "orderParticles": 9})

    def test_chapter_five_demos_keep_independent_returned_controllers(self):
        data = self.run_demos([
            REPO_DIR / "book/assets/demos/physics09.js",
            REPO_DIR / "book/assets/demos/physics10.js",
        ], """
            const assert = require('node:assert/strict');
            const ropeRoot = document.getElementById('pbd-rope');
            const orderRoot = document.getElementById('constraint-order');
            const rope = mounts[0];
            const order = mounts[1];
            assert.equal(mounts.length, 2);
            assert.notEqual(rope.canvas, order.canvas);
            assert.equal(pending.size, 0);
            rope.step();
            assert.equal(rope.getState().frame, 1);
            assert.equal(order.getState().iteration, 0);
            order.step();
            assert.equal(order.getState().iteration, 1);
            assert.equal(rope.getState().frame, 1);
            rope.reset();
            assert.equal(rope.getState().frame, 0);
            assert.equal(order.getState().iteration, 1);
            order.reset();
            assert.equal(order.getState().iteration, 0);
            assert.equal(rope.getState().frame, 0);
            assert.match(readout(ropeRoot).textContent, /最大相对约束误差/);
            assert.match(readout(orderRoot).textContent, /正序最大相对约束误差/);
            console.log(JSON.stringify({independent: true}));
        """)
        self.assertEqual(data, {"independent": True})

    def test_collision_demo_measures_real_shared_plane_and_sphere_contacts(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics11.js"], """
            const assert = require('node:assert/strict');
            assert.equal(mounts.length, 1);
            const demo = mounts[0];
            const root = document.getElementById('collision-friction');
            const toolbar = root.children.find(element => element.className === 'demo-toolbar physics-controls');
            assert.equal(toolbar.children[1].textContent, '运行');
            assert.equal(pending.size, 0);
            assert.deepEqual(inputs(root).map(input => Number(input.value)), [0.5, 0.2, 4, -5]);
            const initial = demo.getState();
            assert.deepEqual(JSON.parse(JSON.stringify(initial.ground.particle)),
                {x: 0, y: 0.2, vx: 4, vy: -5, w: 1});
            assert.deepEqual(JSON.parse(JSON.stringify(initial.circle.particle)),
                {x: 0.36, y: 0.48, vx: 4, vy: -5, w: 1});
            assert.equal(initial.ground.contacts, 0);
            assert.equal(initial.circle.contacts, 0);
            assert.match(demo.canvas.attributes['aria-label'], /地面.*静态圆/);
            demo.step();
            const state = demo.getState();
            assert.equal(planeCalls, 1);
            assert.equal(sphereCalls, 1);
            assert.ok(realPlaneCalls && realSphereCalls);
            assert.equal(state.frame, 1);
            assert.equal(state.time, 1 / 60);
            assert.equal(state.ground.contacts, 1);
            assert.equal(state.circle.contacts, 1);
            assert.deepEqual(JSON.parse(JSON.stringify(state.ground.contact)), {
                correction: 0.3, normalBefore: -5, normalAfter: 2.5,
                tangentBefore: 4, tangentAfter: 3.2
            });
            assert.deepEqual(JSON.parse(JSON.stringify(state.circle.particle)),
                {x: 0.5541333333333334, y: 0.6010666666666667, vx: 4.448, vy: -2.336, w: 1});
            assert.ok(Math.abs(state.circle.contact.correction - 0.2) < 1e-12);
            assert.ok(Math.abs(state.circle.contact.normalBefore + 1.6) < 1e-12);
            assert.ok(Math.abs(state.circle.contact.normalAfter - 0.8) < 1e-12);
            assert.ok(Math.abs(state.circle.contact.tangentBefore - 6.2) < 1e-12);
            assert.ok(Math.abs(state.circle.contact.tangentAfter - 4.96) < 1e-12);
            const firstGroundContact = JSON.parse(JSON.stringify(state.ground.contact));
            const firstCircleContact = JSON.parse(JSON.stringify(state.circle.contact));
            demo.step();
            const next = demo.getState();
            assert.equal(next.frame, 2);
            assert.equal(planeCalls, 2);
            assert.equal(sphereCalls, 2);
            assert.equal(next.ground.contacts, 1);
            assert.equal(next.circle.contacts, 1);
            assert.deepEqual(JSON.parse(JSON.stringify(next.ground.contact)), firstGroundContact);
            assert.deepEqual(JSON.parse(JSON.stringify(next.circle.contact)), firstCircleContact);
            const text = readout(root).textContent;
            for (const marker of ['地面穿透修正量', '静态圆穿透修正量', '地面碰撞前法向速度',
                '地面碰撞后法向速度', '地面碰撞前切向速度', '地面碰撞后切向速度',
                '地面接触次数', '静态圆接触次数']) assert.ok(text.includes(marker), marker);
            assert.doesNotMatch(text, /NaN|Infinity/);
            demo.reset();
            Object.assign(demo.getState().circle.particle, {x: 0, y: 0, vx: 0, vy: 0});
            demo.step();
            const degenerate = demo.getState();
            assert.equal(degenerate.circle.particle.x, 0);
            assert.equal(degenerate.circle.particle.y, 0);
            assert.equal(degenerate.circle.contacts, 0);
            assert.deepEqual(JSON.parse(JSON.stringify(degenerate.circle.contact)), {
                correction: 0, normalBefore: 0, normalAfter: 0,
                tangentBefore: 0, tangentAfter: 0
            });
            console.log(JSON.stringify({ground: state.ground.contact, circle: state.circle.contact,
                groundParticle: state.ground.particle, circleParticle: state.circle.particle}));
        """, setup="""
            const sharedPlane = window.RTPhysics.collideSpherePlane;
            const sharedSphere = window.RTPhysics.collideParticleSphere;
            let planeCalls = 0, sphereCalls = 0, realPlaneCalls = true, realSphereCalls = true;
            window.RTPhysics.collideSpherePlane = (...args) => {
                planeCalls += 1;
                if (args[1] !== 0.5 || args[2] !== 0 || args[0].w !== 1) realPlaneCalls = false;
                return sharedPlane(...args);
            };
            window.RTPhysics.collideParticleSphere = (...args) => {
                sphereCalls += 1;
                if (args[2] !== 0.3 || args[3] !== 0.5 || args[1].x !== 0 || args[1].y !== 0 ||
                    args[0].w !== 1) realSphereCalls = false;
                return sharedSphere(...args);
            };
        """)
        self.assertEqual(data["ground"], {
            "correction": 0.3, "normalBefore": -5, "normalAfter": 2.5,
            "tangentBefore": 4, "tangentAfter": 3.2,
        })
        self.assertAlmostEqual(data["circle"]["correction"], 0.2)
        self.assertAlmostEqual(data["circle"]["normalBefore"], -1.6)
        self.assertAlmostEqual(data["circle"]["normalAfter"], 0.8)
        self.assertAlmostEqual(data["circle"]["tangentBefore"], 6.2)
        self.assertAlmostEqual(data["circle"]["tangentAfter"], 4.96)
        self.assertEqual(data["circleParticle"]["vx"], 4.448)
        self.assertEqual(data["circleParticle"]["vy"], -2.336)

    def test_collision_demo_controls_reset_and_count_only_approaching_contacts(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics11.js"], """
            const assert = require('node:assert/strict');
            const demo = mounts[0];
            const root = document.getElementById('collision-friction');
            const controls = inputs(root);
            const cases = [[0, 2, 4, -5], [0.5, 0.2, 4, 0], [1, 0, -3, 6]];
            for (const values of cases) {
                values.forEach((value, index) => { controls[index].value = String(value); });
                controls[0].dispatch('input');
                let state = demo.getState();
                assert.deepEqual([state.restitution, state.friction, state.vx, state.vy], values);
                assert.equal(state.time, 0);
                assert.equal(state.frame, 0);
                assert.equal(state.ground.contacts, 0);
                assert.equal(state.circle.contacts, 0);
                for (let frame = 0; frame < 120; frame += 1) {
                    demo.step();
                    state = demo.getState();
                    assert.equal(state.finite, true);
                    for (const run of [state.ground, state.circle]) {
                        assert.ok(Number.isFinite(run.contact.correction));
                        assert.ok(Number.isFinite(run.contact.normalBefore));
                        assert.ok(Number.isFinite(run.contact.normalAfter));
                        assert.ok(Number.isFinite(run.contact.tangentBefore));
                        assert.ok(Number.isFinite(run.contact.tangentAfter));
                        for (const key of ['x', 'y', 'vx', 'vy', 'w']) {
                            assert.ok(Number.isFinite(run.particle[key]), key);
                        }
                    }
                }
                assert.equal(state.finished, true);
                assert.equal(state.ground.contacts, values[3] < 0 ? 1 : 0);
                assert.equal(state.circle.contacts, values[3] < 0 ? 1 : 0);
                assert.doesNotMatch(readout(root).textContent, /NaN|Infinity/);
            }
            demo.reset();
            assert.deepEqual(controls.map(input => Number(input.value)), [0.5, 0.2, 4, -5]);
            demo.step();
            const first = JSON.parse(JSON.stringify(demo.getState()));
            demo.reset(); demo.step();
            assert.deepEqual(JSON.parse(JSON.stringify(demo.getState())), first);
            demo.setRunning(true);
            tick(0); tick(1000 / 60);
            assert.equal(demo.getState().frame, 2);
            demo.setRunning(false);
            console.log(JSON.stringify({finite: true, deterministic: true, approachingOnly: true}));
        """)
        self.assertEqual(data, {"finite": True, "deterministic": True, "approachingOnly": True})

    def test_continuous_demo_distinguishes_endpoint_miss_from_exact_swept_times(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics12.js"], """
            const assert = require('node:assert/strict');
            assert.equal(mounts.length, 1);
            const demo = mounts[0];
            const root = document.getElementById('continuous-collision');
            const toolbar = root.children.find(element => element.className === 'demo-toolbar physics-controls');
            assert.equal(toolbar.children[1].textContent, '运行');
            assert.equal(pending.size, 0);
            assert.deepEqual(inputs(root).map(input => Number(input.value)), [18, 0.04, 0.08]);
            let state = demo.getState();
            assert.equal(state.discrete.x, state.swept.x);
            assert.equal(state.speed, 18);
            assert.equal(state.wallThickness, 0.04);
            assert.equal(state.dt, 0.08);
            assert.match(demo.canvas.attributes['aria-label'], /离散.*swept AABB/);
            demo.step();
            state = demo.getState();
            assert.equal(state.time, 0.08);
            assert.equal(state.discrete.x, 1.54);
            assert.equal(state.swept.x, 1.54);
            assert.equal(state.discrete.hit, false);
            assert.equal(state.swept.hit, false);
            assert.equal(state.discrete.missed, false);
            demo.step();
            state = demo.getState();
            assert.equal(state.time, 0.16);
            assert.equal(state.discrete.hit, false);
            assert.equal(state.swept.hit, true);
            assert.equal(state.swept.crossed, true);
            assert.equal(state.postCrossingSteps, 0, 'crossing endpoint is not an additional step');
            assert.equal(state.discrete.missed, false, 'crossing alone must not finalize a miss');
            assert.ok(Math.abs(state.swept.entryTime - 1.7 / 18) < 1e-12);
            assert.ok(Math.abs(state.swept.exitTime - 2.14 / 18) < 1e-12);
            assert.equal(state.swept.hitX, 2);
            assert.equal(state.swept.hitY, 0);
            demo.step();
            state = demo.getState();
            assert.equal(state.frame, 3);
            assert.equal(state.time, 0.24);
            assert.equal(state.swept.crossed, true);
            assert.equal(state.postCrossingSteps, 1);
            assert.equal(state.discrete.missed, false, 'one post-crossing step is still observing');
            demo.step();
            state = demo.getState();
            assert.equal(state.frame, 4);
            assert.equal(state.time, 0.32);
            assert.equal(state.postCrossingSteps, 2);
            assert.equal(state.discrete.hit, false);
            assert.equal(state.discrete.missed, true, 'bounded post-crossing window must finalize the miss');
            assert.equal(state.discrete.x, state.swept.x);
            assert.equal(state.discrete.vx, 18);
            assert.equal(state.swept.vx, 18);
            assert.equal(state.discrete.vy, 0);
            assert.equal(state.swept.vy, 0);
            assert.equal(state.swept.entryTime, 1.7 / 18);
            assert.equal(state.swept.exitTime, 2.14 / 18);
            const text = readout(root).textContent;
            for (const marker of ['离散是否漏碰', '连续进入时间', '连续离开时间', '连续命中位置 x',
                '连续命中位置 y', '物理时间步']) assert.ok(text.includes(marker), marker);
            assert.doesNotMatch(text, /NaN|Infinity/);
            console.log(JSON.stringify({entry: state.swept.entryTime, exit: state.swept.exitTime,
                missed: state.discrete.missed, x: state.discrete.x, time: state.time}));
        """)
        self.assertAlmostEqual(data["entry"], 1.7 / 18)
        self.assertAlmostEqual(data["exit"], 2.14 / 18)
        self.assertTrue(data["missed"])
        self.assertAlmostEqual(data["x"], 5.86)
        self.assertAlmostEqual(data["time"], 0.32)

    def test_continuous_demo_dt_changes_path_and_slower_or_thicker_cases_hit(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics12.js"], """
            const assert = require('node:assert/strict');
            const demo = mounts[0];
            const root = document.getElementById('continuous-collision');
            const controls = inputs(root);
            const secondStepX = {};
            const runCase = (values, name) => {
                values.forEach((value, index) => { controls[index].value = String(value); });
                controls[0].dispatch('input');
                assert.deepEqual([demo.getState().speed, demo.getState().wallThickness, demo.getState().dt], values);
                assert.equal(demo.getState().time, 0);
                assert.equal(demo.getState().frame, 0);
                assert.equal(demo.getState().discrete.hit, false);
                assert.equal(demo.getState().swept.hit, false);
                const frames = Math.ceil(2 / values[2]);
                for (let frame = 0; frame < frames; frame += 1) {
                    demo.step();
                    const state = demo.getState();
                    assert.equal(state.finite, true);
                    assert.equal(state.discrete.x, state.swept.x);
                    if (frame === 1) secondStepX[name] = state.discrete.x;
                    assert.equal(state.discrete.vx, values[0]);
                    assert.equal(state.swept.vx, values[0]);
                    assert.equal(state.discrete.vy, 0);
                    assert.equal(state.swept.vy, 0);
                    for (const value of [state.time, state.discrete.x, state.swept.x,
                        state.swept.entryTime, state.swept.exitTime, state.swept.hitX, state.swept.hitY]) {
                        assert.ok(Number.isFinite(value), 'nonfinite state/readout');
                    }
                    assert.doesNotMatch(readout(root).textContent, /NaN|Infinity/);
                }
                assert.equal(demo.getState().finished, true);
                assert.ok(Math.abs(demo.getState().time - 2) < 1e-12);
                return JSON.parse(JSON.stringify(demo.getState()));
            };
            const slow = runCase([3, 0.04, 0.08], 'slow');
            assert.equal(slow.discrete.hit, true);
            assert.equal(slow.discrete.missed, false);
            const thick = runCase([12, 0.6, 0.08], 'thick');
            assert.equal(thick.discrete.hit, true);
            const fine = runCase([18, 0.04, 0.005], 'fine');
            assert.equal(fine.discrete.hit, true, 'fine dt must change endpoint samples and catch the wall');
            const coarse = runCase([18, 0.04, 0.08], 'coarse');
            assert.equal(coarse.discrete.hit, false);
            assert.equal(coarse.discrete.missed, true);
            assert.equal(secondStepX.fine, 0.28);
            assert.equal(secondStepX.coarse, 2.98);
            assert.notEqual(secondStepX.fine, secondStepX.coarse);
            const stopped = runCase([0, 0.04, 0.2], 'stopped');
            assert.equal(stopped.discrete.hit, false);
            assert.equal(stopped.discrete.missed, false);
            demo.reset();
            assert.deepEqual(controls.map(input => Number(input.value)), [18, 0.04, 0.08]);
            demo.step();
            const first = JSON.parse(JSON.stringify(demo.getState()));
            demo.reset(); demo.step();
            assert.deepEqual(JSON.parse(JSON.stringify(demo.getState())), first);
            demo.reset();
            demo.setRunning(true);
            tick(0); tick(1000 / 60);
            assert.equal(demo.getState().frame, 1, 'UI transport remains 1/60 while motion uses selected dt');
            assert.equal(demo.getState().time, 0.08);
            demo.setRunning(false);
            console.log(JSON.stringify({slowHit: slow.discrete.hit, thickHit: thick.discrete.hit,
                fineHit: fine.discrete.hit, coarseMissed: coarse.discrete.missed, deterministic: true}));
        """)
        self.assertEqual(data, {"slowHit": True, "thickHit": True, "fineHit": True,
                                "coarseMissed": True, "deterministic": True})

    def test_chapter_six_demos_have_independent_controllers_and_fit_narrow_canvas(self):
        data = self.run_demos([
            REPO_DIR / "book/assets/demos/physics11.js",
            REPO_DIR / "book/assets/demos/physics12.js",
        ], """
            const assert = require('node:assert/strict');
            const collision = mounts[0], continuous = mounts[1];
            assert.equal(mounts.length, 2);
            assert.notEqual(collision.canvas, continuous.canvas);
            assert.equal(pending.size, 0);
            collision.step();
            assert.equal(continuous.getState().frame, 0);
            continuous.step();
            assert.equal(collision.getState().frame, 1);
            collision.reset();
            assert.equal(collision.getState().frame, 0);
            assert.equal(continuous.getState().frame, 1);
            continuous.reset();
            assert.equal(continuous.getState().frame, 0);
            for (const controller of [collision, continuous]) {
                controller.canvas.clientWidth = 328;
                controller.canvas.clientHeight = 180;
                controller.canvas.pathDraws = [];
                controller.canvas.textDraws = [];
                controller.draw();
                for (const call of controller.canvas.pathDraws) {
                    assert.ok(Number.isFinite(call.x) && Number.isFinite(call.y));
                    assert.ok(call.x >= 0 && call.x <= 328, 'path x outside narrow canvas');
                    assert.ok(call.y >= 0 && call.y <= 180, 'path y outside narrow canvas');
                }
                for (const call of controller.canvas.textDraws) {
                    assert.ok(call.x >= 0);
                    assert.ok(call.x + call.text.length * 13 <= 328, 'text exceeds narrow canvas: ' + call.text);
                    assert.ok(call.y >= 0 && call.y <= 180);
                }
            }
            assert.match(readout(document.getElementById('collision-friction')).textContent, /接触次数/);
            assert.match(readout(document.getElementById('continuous-collision')).textContent, /离散是否漏碰/);
            collision.setRunning(true); continuous.setRunning(true);
            assert.equal(pending.size, 2);
            collision.reset();
            assert.equal(pending.size, 1);
            tick(0); tick(1000 / 60);
            assert.equal(collision.getState().frame, 0);
            assert.equal(continuous.getState().frame, 1);
            continuous.reset();
            console.log(JSON.stringify({independent: true, width: 328, height: 180}));
        """)
        self.assertEqual(data, {"independent": True, "width": 328, "height": 180})

    def test_chapter_two_demos_mount_independent_controllers(self):
        data = self.run_demos([
            REPO_DIR / "book/assets/demos/physics03.js",
            REPO_DIR / "book/assets/demos/physics04.js",
        ], """
            const assert = require('node:assert/strict');
            const raceRoot = document.getElementById('integration-race');
            const fixedRoot = document.getElementById('fixed-step');
            const race = mounts[0];
            const fixed = mounts[1];
            assert.equal(mounts.length, 2);
            assert.notEqual(race.canvas, fixed.canvas);
            race.step();
            assert.equal(race.getState().frame, 1);
            assert.equal(fixed.getState().frame, 0);
            fixed.step();
            assert.equal(fixed.getState().frame, 1);
            assert.equal(race.getState().frame, 1);
            race.reset();
            assert.equal(race.getState().frame, 0);
            assert.equal(fixed.getState().frame, 1);
            fixed.reset();
            assert.equal(fixed.getState().frame, 0);
            assert.match(readout(raceRoot).textContent, /终点/);
            assert.match(readout(fixedRoot).textContent, /子步数/);
            console.log(JSON.stringify({ok: true, independent: true}));
        """)
        self.assertEqual(data, {"ok": True, "independent": True})

    def test_chapter_demos_mount_independent_controllers(self):
        data = self.run_demos([
            REPO_DIR / "book/assets/demos/physics01.js",
            REPO_DIR / "book/assets/demos/physics02.js",
        ], """
            const assert = require('node:assert/strict');
            const first = document.getElementById('particle-lab');
            const second = document.getElementById('frame-states');
            const firstCanvas = first.children.find(element => element.tagName === 'canvas');
            const secondCanvas = second.children.find(element => element.tagName === 'canvas');
            assert.equal(mounts.length, 2);
            const lab = mounts[0];
            const frames = mounts[1];
            const labToolbar = first.children.find(element => element.className === 'demo-toolbar physics-controls');
            assert.equal(labToolbar.children[1].textContent, '运行');
            frames.setRunning(false);
            lab.step();
            assert.equal(lab.getState().frame, 1);
            assert.equal(frames.getState().time, 0);
            lab.reset();
            assert.equal(lab.getState().frame, 0);
            assert.equal(frames.getState().time, 0);
            assert.match(readout(first).textContent, /下一帧位置/);
            assert.match(readout(second).textContent, /插值误差/);
            console.log(JSON.stringify({ok: true, independent: true}));
        """)
        self.assertEqual(data, {"ok": True, "independent": True})

    def test_xpbd_compliance_compares_real_pbd_and_xpbd_with_per_substep_lambda(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics13.js"], """
            const assert = require('node:assert/strict');
            const demo = mounts[0];
            const root = document.getElementById('xpbd-compliance');
            const toolbar = root.children.find(element => element.className === 'demo-toolbar physics-controls');
            assert.equal(toolbar.children[1].textContent, '运行');
            assert.equal(pending.size, 0);
            assert.deepEqual(inputs(root).map(input => Number(input.value)), [0.01, 2, 4]);
            let state = demo.getState();
            assert.deepEqual(JSON.parse(JSON.stringify(state.pbd.a)),
                {x: 0, y: 0, vx: 0, vy: 0, w: 1});
            assert.deepEqual(JSON.parse(JSON.stringify(state.pbd.b)),
                {x: 3, y: 0, vx: 0, vy: 0, w: 1});
            assert.deepEqual(JSON.parse(JSON.stringify(state.xpbd.a)), JSON.parse(JSON.stringify(state.pbd.a)));
            assert.deepEqual(JSON.parse(JSON.stringify(state.xpbd.b)), JSON.parse(JSON.stringify(state.pbd.b)));
            assert.notEqual(state.pbd.a, state.xpbd.a);
            assert.notEqual(state.pbd.b, state.xpbd.b);
            assert.equal(state.restLength, 2);
            assert.equal(state.xpbd.lambda, 0);
            assert.equal(state.maxRelativeError, 0.5);
            assert.match(demo.canvas.attributes['aria-label'], /PBD.*XPBD.*距离/);

            demo.step();
            state = demo.getState();
            assert.equal(pbdCalls, 8);
            assert.equal(xpbdCalls, 8);
            assert.ok(pbdCallsReal && xpbdCallsReal && lambdaHistoryReal);
            assert.equal(state.frame, 1);
            assert.ok(Math.abs(state.time - 1 / 60) < 1e-12);
            assert.equal(state.h, 1 / 120);
            assert.equal(state.alphaTilde, 144);
            assert.equal(state.pbd.error, 0);
            assert.ok(state.xpbd.error > 0 && state.xpbd.error < 0.5);
            assert.ok(Number.isFinite(state.xpbd.lambda) && state.xpbd.lambda !== 0);
            assert.equal(state.maxRelativeError, Math.max(state.pbd.error, state.xpbd.error));
            assert.ok(Math.abs(state.xpbd.length - 2) > 0);
            const text = readout(root).textContent;
            for (const marker of ['有效柔度', '当前 lambda', '最终长度', '最大相对距离误差',
                                  'PBD', 'XPBD', '模拟时间']) assert.ok(text.includes(marker), marker);
            assert.doesNotMatch(text, /NaN|Infinity/);

            const controls = inputs(root);
            controls[0].value = '0'; controls[0].dispatch('input');
            assert.equal(demo.getState().frame, 0);
            assert.equal(demo.getState().xpbd.lambda, 0);
            demo.step();
            state = demo.getState();
            assert.deepEqual(JSON.parse(JSON.stringify(state.pbd.a)), JSON.parse(JSON.stringify(state.xpbd.a)));
            assert.deepEqual(JSON.parse(JSON.stringify(state.pbd.b)), JSON.parse(JSON.stringify(state.xpbd.b)));
            assert.equal(state.pbd.error, state.xpbd.error);
            assert.equal(state.alphaTilde, 0);
            assert.ok(Math.abs(state.xpbd.lambda) < 1e-12,
                'the final zero-error substep restarts lambda at zero');
            console.log(JSON.stringify({pbdCalls, xpbdCalls, lambdaHistoryReal,
                softLength: state.xpbd.length, lambda: state.xpbd.lambda, zeroComplianceMatches: true}));
        """, setup="""
            const sharedPbd = window.RTPhysics.projectDistance;
            const sharedXpbd = window.RTPhysics.projectDistanceXPBD;
            let pbdCalls = 0, xpbdCalls = 0, pbdCallsReal = true, xpbdCallsReal = true;
            let lambdaHistoryReal = true;
            let expectedLambda = 0;
            const xpbdHistory = [];
            window.RTPhysics.projectDistance = (...args) => {
                pbdCalls += 1;
                if (args[2] !== 2 || args[3] !== 1) pbdCallsReal = false;
                return sharedPbd(...args);
            };
            window.RTPhysics.projectDistanceXPBD = (...args) => {
                xpbdCalls += 1;
                if (args[2] !== 2 || args[4] !== 1 / 120) xpbdCallsReal = false;
                if (args[5] !== expectedLambda) lambdaHistoryReal = false;
                const result = sharedXpbd(...args);
                xpbdHistory.push({input: args[5], result: result.lambda});
                expectedLambda = xpbdCalls % 4 === 0 ? 0 : result.lambda;
                return result;
            };
        """)
        self.assertEqual(data["pbdCalls"], 16)
        self.assertEqual(data["xpbdCalls"], 16)
        self.assertTrue(data["lambdaHistoryReal"])
        self.assertTrue(data["zeroComplianceMatches"])

    def test_xpbd_compliance_substeps_change_projection_and_controls_reset_deterministically(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics13.js"], """
            const assert = require('node:assert/strict');
            const demo = mounts[0];
            const root = document.getElementById('xpbd-compliance');
            const controls = inputs(root);
            const results = {};
            for (const substeps of [1, 4]) {
                controls[0].value = '0.01'; controls[1].value = String(substeps);
                controls[2].value = '1'; controls[0].dispatch('input');
                const callsBeforeStep = xpbdCalls;
                demo.step();
                const state = demo.getState();
                assert.equal(xpbdCalls - callsBeforeStep, substeps,
                    'substep and iteration controls must drive real XPBD projections');
                assert.equal(state.h, 1 / (60 * substeps));
                assert.equal(state.alphaTilde, 0.01 / (state.h * state.h));
                assert.ok(Math.abs(state.time - 1 / 60) < 1e-12);
                assert.ok(state.xpbd.length > 2 && state.xpbd.length < 3);
                results[substeps] = {length: state.xpbd.length, error: state.xpbd.error,
                    alphaTilde: state.alphaTilde};
            }
            assert.notEqual(results[1].length, results[4].length);
            assert.notEqual(results[1].error, results[4].error);
            assert.ok(Math.abs(results[1].alphaTilde - 36) < 1e-9);
            assert.ok(Math.abs(results[4].alphaTilde - 576) < 1e-9);

            const callsBeforeIterationCase = xpbdCalls;
            controls[1].value = '2'; controls[2].value = '8'; controls[1].dispatch('input');
            demo.step();
            const converged = demo.getState();
            assert.equal(xpbdCalls - callsBeforeIterationCase, 16,
                'iteration control must drive real XPBD projections');
            assert.ok(Number.isFinite(converged.xpbd.error));
            assert.equal(converged.pbd.error, 0);
            demo.reset();
            assert.deepEqual(inputs(root).map(input => Number(input.value)), [0.01, 2, 4]);
            assert.equal(demo.getState().xpbd.lambda, 0);
            demo.step();
            const first = JSON.parse(JSON.stringify(demo.getState()));
            demo.reset(); demo.step();
            assert.deepEqual(JSON.parse(JSON.stringify(demo.getState())), first);

            demo.reset();
            corruptNextProjection = true;
            demo.step();
            const invalid = demo.getState();
            assert.equal(invalid.invalid, true);
            assert.equal(invalid.finished, true);
            assert.equal(invalid.finite, false);
            assert.equal(pending.size, 0);
            const warning = children(root).find(element => element.attributes.role === 'status');
            assert.equal(warning.hidden, false);
            assert.match(warning.textContent, /非有限/);
            assert.match(warning.textContent, /重置/);
            assert.doesNotMatch(readout(root).textContent, /NaN|Infinity/);
            const stopped = JSON.parse(JSON.stringify(demo.getState()));
            demo.step();
            assert.deepEqual(JSON.parse(JSON.stringify(demo.getState())), stopped);
            demo.reset();
            assert.equal(demo.getState().invalid, false);
            assert.equal(warning.hidden, true);
            console.log(JSON.stringify({results, convergedError: converged.xpbd.error,
                deterministic: true, invalidPaused: true}));
        """, setup="""
            const sharedXpbd = window.RTPhysics.projectDistanceXPBD;
            let xpbdCalls = 0, corruptNextProjection = false;
            window.RTPhysics.projectDistanceXPBD = (...args) => {
                xpbdCalls += 1;
                const result = sharedXpbd(...args);
                if (corruptNextProjection) {
                    corruptNextProjection = false;
                    result.a.x = Number.NaN;
                }
                return result;
            };
        """)
        self.assertAlmostEqual(data["results"]["1"]["alphaTilde"], 36)
        self.assertAlmostEqual(data["results"]["4"]["alphaTilde"], 576)
        self.assertNotEqual(data["results"]["4"]["error"], data["results"]["1"]["error"])
        self.assertTrue(data["deterministic"])
        self.assertTrue(data["invalidPaused"])

    def test_area_constraint_uses_real_projection_lambda_history_and_finite_metrics(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics14.js"], """
            const assert = require('node:assert/strict');
            const demo = mounts[0];
            const root = document.getElementById('area-constraint');
            assert.deepEqual(inputs(root).map(input => Number(input.value)), [0.01, 8]);
            let state = demo.getState();
            assert.deepEqual(JSON.parse(JSON.stringify(state.particles)), [
                {x: 0, y: 0, vx: 0, vy: 0, w: 1},
                {x: 3, y: 0, vx: 0, vy: 0, w: 1},
                {x: 3, y: 1, vx: 0, vy: 0, w: 1},
                {x: 0, y: 1, vx: 0, vy: 0, w: 1},
            ]);
            assert.equal(state.restArea, 2);
            assert.equal(state.area, 3);
            assert.equal(state.relativeError, 0.5);
            assert.equal(state.lambda, 0);
            assert.equal(state.maxAbsLambda, 0);
            assert.match(demo.canvas.attributes['aria-label'], /四边形.*面积.*拖拽/);
            demo.step();
            state = demo.getState();
            assert.equal(areaCalls, 8);
            assert.ok(realCalls && lambdaHistoryReal);
            assert.equal(state.frame, 1);
            assert.ok(Math.abs(state.time - 1 / 60) < 1e-12);
            assert.ok(state.area > 2 && state.area < 3);
            assert.ok(state.relativeError >= 0 && state.relativeError < 0.5);
            assert.ok(Number.isFinite(state.lambda) && state.lambda !== 0);
            assert.ok(state.maxAbsLambda >= Math.abs(state.lambda));
            assert.ok(state.alphaTilde === 36);
            const text = readout(root).textContent;
            for (const marker of ['当前面积', '静止面积', '相对面积误差', '当前 lambda',
                                  '最大乘子绝对值', '有效柔度']) assert.ok(text.includes(marker), marker);
            assert.doesNotMatch(text, /NaN|Infinity/);
            const values = children(root).filter(element => element.className === 'metric')
                .map(element => Number(element.children[1].textContent));
            assert.ok(values.length >= 8 && values.every(Number.isFinite));

            const softRelativeError = state.relativeError;
            const softSnapshot = JSON.parse(JSON.stringify(state));
            areaCalls = 0;
            const controls = inputs(root);
            controls[0].value = '0'; controls[0].dispatch('input');
            assert.equal(demo.getState().frame, 0);
            assert.equal(demo.getState().area, 3);
            assert.equal(demo.getState().lambda, 0);
            previousResult = 0;
            demo.step();
            assert.ok(demo.getState().relativeError < softRelativeError);
            console.log(JSON.stringify({areaCalls, lambdaHistoryReal, area: softSnapshot.area,
                relativeError: softSnapshot.relativeError, lambda: softSnapshot.lambda,
                maxAbsLambda: softSnapshot.maxAbsLambda}));
        """, setup="""
            const sharedArea = window.RTPhysics.projectAreaXPBD;
            let areaCalls = 0, realCalls = true, lambdaHistoryReal = true;
            let previousResult = 0;
            window.RTPhysics.projectAreaXPBD = (...args) => {
                areaCalls += 1;
                if (args[4] !== 2 || args[5] !== 0.01 || args[6] !== 1 / 60) realCalls = false;
                if (args[7] !== previousResult) lambdaHistoryReal = false;
                const result = sharedArea(...args);
                previousResult = result.lambda;
                return result;
            };
        """)
        self.assertEqual(data["areaCalls"], 8)
        self.assertTrue(data["lambdaHistoryReal"])
        self.assertGreater(data["area"], 2)
        self.assertLess(data["area"], 3)
        self.assertLess(data["relativeError"], 0.5)

    def test_area_constraint_drag_pauses_clamps_and_rejects_degenerate_or_nonfinite_geometry(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics14.js"], """
            const assert = require('node:assert/strict');
            const demo = mounts[0];
            const root = document.getElementById('area-constraint');
            const canvas = demo.canvas;
            canvas.clientWidth = 328; canvas.clientHeight = 180;
            demo.step();
            assert.notEqual(demo.getState().lambda, 0);
            demo.setRunning(true);
            assert.equal(pending.size, 1);
            canvas.pathDraws = [];
            demo.draw();
            const points = canvas.pathDraws.filter(call => call.op === 'arc');
            assert.equal(points.length, 4);
            canvas.dispatch('pointerdown', {clientX: points[0].x, clientY: points[0].y, pointerId: 7});
            assert.equal(pending.size, 0);
            assert.equal(demo.getState().lambda, 0);
            assert.equal(demo.getState().maxAbsLambda, 0);
            canvas.dispatch('pointermove', {clientX: -1000, clientY: -1000, pointerId: 7});
            canvas.dispatch('pointerup', {pointerId: 7});
            let dragged = demo.getState().particles[0];
            assert.ok(dragged.x >= -0.5 && dragged.x <= 3.5);
            assert.ok(dragged.y >= -0.5 && dragged.y <= 1.5);
            assert.equal(dragged.vx, 0); assert.equal(dragged.vy, 0);
            assert.equal(demo.getState().dragging, null);
            assert.ok(Math.abs(demo.getState().area) > 1e-8);
            demo.step();
            assert.ok(firstLambdaAfterDrag === 0);

            demo.reset();
            canvas.pathDraws = []; demo.draw();
            const resetPoints = canvas.pathDraws.filter(call => call.op === 'arc');
            demo.step();
            canvas.pathDraws = []; demo.draw();
            const currentPoints = canvas.pathDraws.filter(call => call.op === 'arc');
            canvas.dispatch('pointerdown', {clientX: currentPoints[1].x, clientY: currentPoints[1].y, pointerId: 8});
            const beforeInvalid = JSON.parse(JSON.stringify(demo.getState().particles));
            canvas.dispatch('pointermove', {clientX: currentPoints[0].x, clientY: currentPoints[0].y, pointerId: 8});
            canvas.dispatch('pointerup', {pointerId: 8});
            let invalid = demo.getState();
            assert.equal(invalid.invalid, true); assert.equal(invalid.finished, true);
            assert.deepEqual(JSON.parse(JSON.stringify(invalid.particles)), beforeInvalid);
            assert.equal(pending.size, 0);
            const warning = children(root).find(element => element.attributes.role === 'status');
            assert.equal(warning.hidden, false);
            assert.match(warning.textContent, /重合|共线|退化/);
            assert.match(warning.textContent, /重置/);
            assert.doesNotMatch(readout(root).textContent, /NaN|Infinity/);

            demo.reset();
            canvas.pathDraws = []; demo.draw();
            const collinearPoints = canvas.pathDraws.filter(call => call.op === 'arc');
            const midpoint = {
                x: (collinearPoints[0].x + collinearPoints[2].x) / 2,
                y: (collinearPoints[0].y + collinearPoints[2].y) / 2,
            };
            assert.equal((collinearPoints[2].x - collinearPoints[0].x) * (midpoint.y - collinearPoints[0].y),
                (collinearPoints[2].y - collinearPoints[0].y) * (midpoint.x - collinearPoints[0].x));
            canvas.dispatch('pointerdown', {clientX: collinearPoints[1].x, clientY: collinearPoints[1].y, pointerId: 10});
            const beforeCollinear = JSON.parse(JSON.stringify(demo.getState().particles));
            canvas.dispatch('pointermove', {clientX: midpoint.x, clientY: midpoint.y, pointerId: 10});
            canvas.dispatch('pointerup', {pointerId: 10});
            invalid = demo.getState();
            assert.equal(invalid.invalid, true); assert.equal(invalid.finished, true);
            assert.equal(invalid.finite, false);
            assert.deepEqual(JSON.parse(JSON.stringify(invalid.particles)), beforeCollinear);
            assert.equal(pending.size, 0);
            assert.equal(warning.hidden, false);
            assert.match(warning.textContent, /共线|退化/);
            assert.match(warning.textContent, /重置/);
            assert.doesNotMatch(readout(root).textContent, /NaN|Infinity/);

            demo.reset();
            canvas.pathDraws = []; demo.draw();
            const reversedPoints = canvas.pathDraws.filter(call => call.op === 'arc');
            canvas.dispatch('pointerdown', {clientX: reversedPoints[0].x, clientY: reversedPoints[0].y, pointerId: 11});
            const beforeReversal = JSON.parse(JSON.stringify(demo.getState().particles));
            const scale = reversedPoints[2].y - reversedPoints[1].y;
            const left = reversedPoints[1].x - 3 * scale;
            const top = reversedPoints[0].y;
            canvas.dispatch('pointermove', {clientX: left + 4 * scale, clientY: top + 2 * scale, pointerId: 11});
            canvas.dispatch('pointerup', {pointerId: 11});
            invalid = demo.getState();
            assert.equal(invalid.invalid, true); assert.equal(invalid.finished, true);
            assert.equal(invalid.finite, false);
            assert.deepEqual(JSON.parse(JSON.stringify(invalid.particles)), beforeReversal);
            assert.equal(pending.size, 0);
            assert.equal(warning.hidden, false);
            assert.match(warning.textContent, /翻面|退化/);
            assert.match(warning.textContent, /重置/);
            assert.doesNotMatch(readout(root).textContent, /NaN|Infinity/);

            demo.reset();
            canvas.pathDraws = []; demo.draw();
            const finitePoints = canvas.pathDraws.filter(call => call.op === 'arc');
            canvas.dispatch('pointerdown', {clientX: finitePoints[0].x, clientY: finitePoints[0].y, pointerId: 9});
            const finiteBefore = JSON.parse(JSON.stringify(demo.getState().particles));
            canvas.dispatch('pointermove', {clientX: Number.NaN, clientY: 20, pointerId: 9});
            canvas.dispatch('pointerup', {pointerId: 9});
            invalid = demo.getState();
            assert.equal(invalid.invalid, true);
            assert.deepEqual(JSON.parse(JSON.stringify(invalid.particles)), finiteBefore);
            assert.match(warning.textContent, /非有限/);
            assert.match(warning.textContent, /重置/);
            assert.doesNotMatch(readout(root).textContent, /NaN|Infinity/);

            demo.reset();
            assert.equal(demo.getState().area, 3);
            assert.equal(demo.getState().lambda, 0);
            assert.equal(demo.getState().dragging, null);
            assert.equal(warning.hidden, true);
            const controls = inputs(root);
            controls[0].value = '0.5'; controls[1].value = '3'; controls[0].dispatch('input');
            assert.equal(demo.getState().compliance, 0.5);
            assert.equal(demo.getState().iterations, 3);
            assert.equal(demo.getState().area, 3);
            assert.equal(demo.getState().lambda, 0);
            console.log(JSON.stringify({dragPaused: true, clamped: true, lambdaReset: true,
                degeneratePaused: true, collinearPaused: true, reversedPaused: true,
                nonfinitePaused: true}));
        """, setup="""
            const sharedArea = window.RTPhysics.projectAreaXPBD;
            let firstLambdaAfterDrag = null;
            let nextUpdateIsAfterDrag = false;
            window.RTPhysics.projectAreaXPBD = (...args) => {
                if (nextUpdateIsAfterDrag && firstLambdaAfterDrag === null) firstLambdaAfterDrag = args[7];
                nextUpdateIsAfterDrag = false;
                return sharedArea(...args);
            };
            document.getElementById('area-constraint').addEventListener('pointerup', () => {
                nextUpdateIsAfterDrag = true;
            });
        """)
        self.assertEqual(data, {"dragPaused": True, "clamped": True, "lambdaReset": True,
                                "degeneratePaused": True, "collinearPaused": True,
                                "reversedPaused": True, "nonfinitePaused": True})

    def test_chapter_seven_demos_are_independent_reduced_motion_safe_and_fit_narrow_canvas(self):
        data = self.run_demos([
            REPO_DIR / "book/assets/demos/physics13.js",
            REPO_DIR / "book/assets/demos/physics14.js",
        ], """
            const assert = require('node:assert/strict');
            const compliance = mounts[0], area = mounts[1];
            const complianceRoot = document.getElementById('xpbd-compliance');
            const areaRoot = document.getElementById('area-constraint');
            assert.equal(mounts.length, 2);
            assert.notEqual(compliance.canvas, area.canvas);
            assert.equal(pending.size, 0, 'reduced motion must leave both demos paused');
            compliance.step();
            assert.equal(area.getState().frame, 0);
            area.step();
            assert.equal(compliance.getState().frame, 1);
            for (const controller of [compliance, area]) {
                controller.canvas.clientWidth = 328;
                controller.canvas.clientHeight = 180;
                controller.canvas.pathDraws = [];
                controller.canvas.textDraws = [];
                controller.draw();
                assert.equal(controller.canvas.width, 328);
                assert.equal(controller.canvas.height, 180);
                for (const call of controller.canvas.pathDraws) {
                    assert.ok(Number.isFinite(call.x) && Number.isFinite(call.y));
                    assert.ok(call.x >= 0 && call.x <= 328, 'path x outside 328px canvas');
                    assert.ok(call.y >= 0 && call.y <= 180, 'path y outside 180px canvas');
                }
                for (const call of controller.canvas.textDraws) {
                    assert.ok(Number.isFinite(call.x) && Number.isFinite(call.y));
                    assert.ok(call.x >= 0 && call.x + call.text.length * 13 <= 328, call.text);
                    assert.ok(call.y >= 0 && call.y <= 180, call.text);
                }
            }
            assert.equal(compliance.canvas.pathDraws.filter(call => call.op === 'arc').length, 4);
            assert.equal(area.canvas.pathDraws.filter(call => call.op === 'arc').length, 4);
            assert.match(readout(complianceRoot).textContent, /有效柔度/);
            assert.match(readout(areaRoot).textContent, /相对面积误差/);
            compliance.setRunning(true); area.setRunning(true);
            assert.equal(pending.size, 2);
            compliance.reset();
            assert.equal(pending.size, 1);
            tick(0); tick(1000 / 60);
            assert.equal(compliance.getState().frame, 0);
            assert.equal(area.getState().frame, 2);
            area.reset();
            assert.equal(pending.size, 0);
            console.log(JSON.stringify({independent: true, reducedMotion: true, width: 328, height: 180}));
        """, setup="""
            window.matchMedia = () => ({matches: true});
        """)
        self.assertEqual(data, {"independent": True, "reducedMotion": True,
                                "width": 328, "height": 180})

    def test_area_constraint_bounds_iterations_and_finishes_with_finite_readouts(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics14.js"], """
            const assert = require('node:assert/strict');
            const demo = mounts[0];
            const root = document.getElementById('area-constraint');
            const controls = inputs(root);
            controls[0].value = '1'; controls[1].value = '40'; controls[0].dispatch('input');
            for (let frame = 0; frame < 60; frame += 1) {
                const callsBeforeStep = areaCalls;
                demo.step();
                assert.equal(areaCalls - callsBeforeStep, 40,
                    'iteration control must drive 40 real area projections per update');
                const state = demo.getState();
                assert.equal(state.finite, true);
                assert.ok(Math.abs(state.area) > 1e-8);
                for (const value of [state.area, state.restArea, state.relativeError,
                    state.lambda, state.maxAbsLambda, state.alphaTilde, state.time]) {
                    assert.ok(Number.isFinite(value));
                }
                assert.doesNotMatch(readout(root).textContent, /NaN|Infinity/);
            }
            let state = demo.getState();
            assert.equal(state.frame, 60);
            assert.equal(areaCalls, 2400);
            const callsAt60Frames = areaCalls;
            assert.equal(state.finished, true);
            assert.ok(Math.abs(state.time - 1) < 1e-12);
            const stopped = JSON.parse(JSON.stringify(state));
            demo.step();
            assert.deepEqual(JSON.parse(JSON.stringify(demo.getState())), stopped);
            const warning = children(root).find(element => element.attributes.role === 'status');
            assert.equal(warning.hidden, false);
            assert.match(warning.textContent, /重置/);
            demo.reset();
            assert.deepEqual(inputs(root).map(input => Number(input.value)), [0.01, 8]);
            assert.equal(demo.getState().frame, 0);
            assert.equal(demo.getState().finished, false);
            demo.step();
            const first = JSON.parse(JSON.stringify(demo.getState()));
            demo.reset(); demo.step();
            assert.deepEqual(JSON.parse(JSON.stringify(demo.getState())), first);
            console.log(JSON.stringify({frames: state.frame, callsAt60Frames, finite: true,
                deterministic: true, area: state.area, error: state.relativeError}));
        """, setup="""
            const sharedArea = window.RTPhysics.projectAreaXPBD;
            let areaCalls = 0;
            window.RTPhysics.projectAreaXPBD = (...args) => {
                areaCalls += 1;
                return sharedArea(...args);
            };
        """)
        self.assertEqual(data["frames"], 60)
        self.assertEqual(data["callsAt60Frames"], 2400)
        self.assertTrue(data["finite"])
        self.assertTrue(data["deterministic"])


class PhysicsChapterEightDemoTests(unittest.TestCase):
    run_demos = PhysicsChapterDemoTests.run_demos

    def test_solver_race_uses_real_core_equal_initial_states_and_reconstructed_energy(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics15.js"], """
            const assert = require('node:assert/strict');
            const demo = mounts[0];
            const root = document.getElementById('solver-race');
            assert.equal(toolbarText(), '运行');
            assert.equal(pending.size, 0);
            assert.deepEqual(inputs(root).map(input => Number(input.value)), [1, 2, 8]);
            let state = demo.getState();
            for (const name of ['massSpring', 'pbd', 'xpbd']) {
                assert.deepEqual(JSON.parse(JSON.stringify(state[name].a)),
                    {x: -1.5, y: 0, vx: -0.6, vy: 0, w: 1});
                assert.deepEqual(JSON.parse(JSON.stringify(state[name].b)),
                    {x: 1.5, y: 0, vx: 0.6, vy: 0, w: 1});
            }
            assert.notEqual(state.massSpring.a, state.pbd.a);
            assert.notEqual(state.pbd.a, state.xpbd.a);
            assert.equal(state.stiffness, 120);
            assert.equal(state.compliance, 0.01);
            assert.equal(state.outerDt, 1 / 60);
            assert.equal(state.reducedMass, 0.5);
            assert.ok(Math.abs(state.stableStep - 2 / Math.sqrt(120 / 0.5)) < 1e-12);
            assert.equal(state.xpbd.lambda, 0);
            assert.match(demo.canvas.attributes['aria-label'], /质量弹簧.*PBD.*XPBD/);
            const beforeSpring = JSON.parse(JSON.stringify(state.massSpring));
            demo.step();
            state = demo.getState();
            assert.equal(state.frame, 1);
            assert.ok(Math.abs(state.time - 1 / 60) < 1e-12);
            assert.equal(springCalls, 1);
            assert.equal(stableCalls, 2);
            assert.equal(pbdCalls, 8);
            assert.equal(xpbdCalls, 8);
            assert.ok(realCalls && lambdaHistoryReal && stableForced);
            assert.deepEqual(springArguments, [[1, 1.2, 120, 0]]);
            assert.deepEqual(springResults, [-120]);
            const deltaA = state.massSpring.a.vx - beforeSpring.a.vx;
            const deltaB = state.massSpring.b.vx - beforeSpring.b.vx;
            assert.equal(deltaA, 2);
            assert.equal(deltaB, -2);
            assert.ok(Math.abs(deltaA + deltaB) < 1e-12,
                'equal masses receive equal and opposite axial velocity changes');
            assert.ok(state.massSpring.length < beforeSpring.length,
                'an initially stretched, outwardly moving pair must shorten on its first step');
            assert.ok(Math.abs(state.massSpring.length - (3 - 2.8 / 60)) < 1e-12);
            assert.equal(state.massSpring.a.vx, 1.4);
            assert.equal(state.massSpring.b.vx, -1.4);
            assert.equal(state.massSpring.a.vy, 0);
            assert.equal(state.massSpring.b.vy, 0);
            assert.ok(Math.abs(state.massSpring.kinetic - 1.96) < 1e-12);
            assert.ok(Math.abs(state.massSpring.elasticPotential - 54.53066666666667) < 1e-10);
            assert.equal(state.pbd.length, 2);
            assert.equal(state.pbd.a.vx, 30);
            assert.equal(state.pbd.b.vx, -30);
            assert.ok(state.xpbd.length > 2.9 && state.xpbd.length < 3,
                'positive compliance retains a visible residual after prediction');
            assert.notEqual(state.xpbd.lambda, 0);
            for (const name of ['massSpring', 'pbd', 'xpbd']) {
                const run = state[name];
                const kinetic = 0.5 * state.mass * (run.a.vx ** 2 + run.a.vy ** 2 +
                    run.b.vx ** 2 + run.b.vy ** 2);
                assert.ok(Math.abs(run.kinetic - kinetic) < 1e-10, name);
                assert.equal(run.maxRelativeError, Math.max(run.maxRelativeError, run.error));
                assert.ok(Number.isFinite(run.maxRelativeError));
            }
            assert.equal(state.massSpring.totalMechanical,
                state.massSpring.kinetic + 0.5 * state.stiffness *
                    (state.massSpring.length - state.restLength) ** 2);
            assert.equal(state.massSpring.kineticDelta, state.massSpring.kinetic - state.initialKinetic);
            const text = readout(root).textContent;
            for (const marker of ['质量弹簧', 'PBD', 'XPBD', '最终长度', '最大相对距离误差',
                                  '动能', '动能变化', '质量弹簧总机械能', '弹性势能', '力积分']) {
                assert.ok(text.includes(marker), marker);
            }
            assert.doesNotMatch(text, /NaN|Infinity/);
            const controls = inputs(root);
            controls[2].value = '1'; controls[2].dispatch('input');
            demo.step();
            state = demo.getState();
            assert.equal(pbdCalls, 9);
            assert.equal(xpbdCalls, 9);
            assert.equal(springCalls, 2, 'iterations do not change force integration count');
            assert.equal(state.pbd.length, 2,
                'one independent PBD distance constraint converges in one projection');
            controls[0].value = '0.5'; controls[1].value = '1.5'; controls[0].dispatch('input');
            state = demo.getState();
            assert.equal(state.frame, 0);
            assert.equal(state.mass, 0.5);
            assert.equal(state.restLength, 1.5);
            assert.equal(state.massSpring.a.w, 2);
            assert.equal(state.initialKinetic, 0.18);
            assert.ok(state.stableStep > state.outerDt);
            console.log(JSON.stringify({realCalls, lambdaHistoryReal, stableForced,
                lengths: [state.massSpring.length, state.pbd.length, state.xpbd.length],
                springCalls, pbdCalls, xpbdCalls}));
        """, setup="""
            const shared = window.RTPhysics;
            const pbdFn = shared.projectDistance, xpbdFn = shared.projectDistanceXPBD;
            const forceFn = shared.springForce, stableFn = shared.stableSpringStep;
            let springCalls = 0, stableCalls = 0, pbdCalls = 0, xpbdCalls = 0;
            let realCalls = true, lambdaHistoryReal = true, stableForced = true;
            let expectedLambda = 0;
            const springArguments = [], springResults = [];
            shared.springForce = (...args) => {
                springCalls += 1;
                springArguments.push(args);
                const result = forceFn(...args);
                springResults.push(result);
                return result;
            };
            shared.stableSpringStep = (...args) => {
                stableCalls += 1;
                if (args[0] !== 120) stableForced = false;
                const dt = stableFn(...args);
                if (!(1 / 60 < dt)) stableForced = false;
                return dt;
            };
            shared.projectDistance = (...args) => {
                pbdCalls += 1;
                if (args[3] !== 1) realCalls = false;
                return pbdFn(...args);
            };
            shared.projectDistanceXPBD = (...args) => {
                xpbdCalls += 1;
                if (args[3] !== 0.01 || args[4] !== 1 / 60) realCalls = false;
                if (args[5] !== expectedLambda) lambdaHistoryReal = false;
                const result = xpbdFn(...args);
                expectedLambda = result.lambda;
                return result;
            };
            document.getElementById('solver-race').addEventListener('input', () => { expectedLambda = 0; });
            function toolbarText() {
                return document.getElementById('solver-race').children.find(
                    element => element.className === 'demo-toolbar physics-controls').children[1].textContent;
            }
        """)
        self.assertTrue(data["realCalls"])
        self.assertTrue(data["lambdaHistoryReal"])
        self.assertTrue(data["stableForced"])
        self.assertEqual((data["springCalls"], data["pbdCalls"], data["xpbdCalls"]), (2, 9, 9))

    def test_solver_race_drawing_fits_328_by_180_during_default_playback(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics15.js"], """
            const assert = require('node:assert/strict');
            const demo = mounts[0];
            demo.canvas.clientWidth = 328; demo.canvas.clientHeight = 180;
            let checkedDraws = 0;
            function assertBoundedDraw(frame) {
                demo.canvas.pathDraws = []; demo.canvas.textDraws = [];
                demo.draw();
                assert.equal(demo.canvas.width, 328);
                assert.equal(demo.canvas.height, 180);
                assert.equal(demo.canvas.pathDraws.filter(call => call.op === 'arc').length, 6);
                for (const call of demo.canvas.pathDraws) {
                    assert.ok(Number.isFinite(call.x) && Number.isFinite(call.y), 'frame ' + frame);
                    assert.ok(call.x >= 0 && call.x <= 328, 'path x outside canvas at frame ' + frame);
                    assert.ok(call.y >= 0 && call.y <= 180, 'path y outside canvas at frame ' + frame);
                }
                for (const call of demo.canvas.textDraws) {
                    assert.ok(Number.isFinite(call.x) && Number.isFinite(call.y));
                    assert.ok(call.x >= 0 && call.x + call.text.length * 13 <= 328,
                        call.text + ' outside canvas at frame ' + frame);
                    assert.ok(call.y >= 0 && call.y <= 180, call.text);
                }
                checkedDraws += 1;
            }
            assertBoundedDraw(0);
            for (let frame = 1; frame <= 60; frame += 1) {
                demo.step();
                assertBoundedDraw(frame);
            }
            const state = demo.getState();
            assert.equal(state.frame, 60);
            assert.equal(state.finished, true);
            assert.equal(state.finite, true);
            console.log(JSON.stringify({fits: true, pairs: 3, checkedDraws,
                finalMassSpringLength: state.massSpring.length}));
        """)
        self.assertEqual(data["fits"], True)
        self.assertEqual(data["pairs"], 3)
        self.assertEqual(data["checkedDraws"], 61)

    def test_solver_race_finishes_resets_and_pauses_on_nonfinite_core_output(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics15.js"], """
            const assert = require('node:assert/strict');
            const demo = mounts[0];
            const root = document.getElementById('solver-race');
            demo.setRunning(true);
            assert.equal(pending.size, 1);
            for (let frame = 0; frame < 60; frame += 1) demo.step();
            let state = demo.getState();
            assert.equal(state.frame, 60);
            assert.equal(state.finished, true);
            assert.equal(state.finite, true);
            assert.ok(Math.abs(state.time - 1) < 1e-12);
            assert.equal(pending.size, 0);
            for (const name of ['massSpring', 'pbd', 'xpbd']) {
                for (const value of [state[name].length, state[name].error, state[name].kinetic,
                    state[name].kineticDelta, state[name].maxRelativeError]) {
                    assert.ok(Number.isFinite(value), name);
                }
            }
            const stopped = JSON.parse(JSON.stringify(state));
            demo.step();
            assert.deepEqual(JSON.parse(JSON.stringify(demo.getState())), stopped);
            demo.reset(); demo.step();
            const first = JSON.parse(JSON.stringify(demo.getState()));
            demo.reset(); demo.step();
            assert.deepEqual(JSON.parse(JSON.stringify(demo.getState())), first);
            demo.reset();
            corrupt = true;
            demo.step();
            state = demo.getState();
            assert.equal(state.invalid, true);
            assert.equal(state.finished, true);
            assert.equal(state.finite, false);
            assert.equal(pending.size, 0);
            const warning = children(root).find(element => element.attributes.role === 'status');
            assert.equal(warning.hidden, false);
            assert.match(warning.textContent, /非有限/);
            assert.match(warning.textContent, /重置/);
            assert.doesNotMatch(readout(root).textContent, /NaN|Infinity/);
            demo.reset();
            assert.equal(demo.getState().finished, false);
            assert.equal(warning.hidden, true);
            console.log(JSON.stringify({finished: true, finite: true, deterministic: true,
                invalidPaused: true, frame: stopped.frame}));
        """, setup="""
            let corrupt = false;
            const xpbdFn = window.RTPhysics.projectDistanceXPBD;
            window.RTPhysics.projectDistanceXPBD = (...args) => {
                const result = xpbdFn(...args);
                if (corrupt) { corrupt = false; result.a.vx = Number.POSITIVE_INFINITY; }
                return result;
            };
        """)
        self.assertEqual(data, {"finished": True, "finite": True, "deterministic": True,
                                "invalidPaused": True, "frame": 60})

    def test_substep_demo_runs_equal_duration_selected_and_reference_with_lambda_lifecycle(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics16.js"], """
            const assert = require('node:assert/strict');
            const demo = mounts[0];
            const root = document.getElementById('substep-invariance');
            assert.deepEqual(inputs(root).map(input => Number(input.value)), [2, 8, 0.01]);
            assert.equal(pending.size, 0);
            let state = demo.getState();
            assert.equal(state.totalDuration, 1);
            assert.equal(state.referenceSubsteps, 8);
            assert.equal(state.selectedSubsteps, 2);
            assert.equal(state.restLength, 2);
            for (const method of ['pbd', 'xpbd']) {
                for (const group of ['selected', 'reference']) {
                    assert.equal(state[method][group].lambda, 0);
                    assert.deepEqual(JSON.parse(JSON.stringify(state[method][group].a)),
                        {x: -1.5, y: 0, vx: -0.6, vy: 0, w: 1});
                }
            }
            assert.notEqual(state.pbd.selected.a, state.pbd.reference.a);
            assert.match(demo.canvas.attributes['aria-label'], /子步.*PBD.*XPBD.*误差条/);
            demo.step();
            state = demo.getState();
            assert.equal(state.frame, 1);
            assert.ok(Math.abs(state.time - 1 / 60) < 1e-12);
            assert.equal(pbdCalls, 80);
            assert.equal(xpbdCalls, 80);
            assert.ok(realCalls && lambdaLifecycle);
            assert.equal(state.selectedH, 1 / 120);
            assert.equal(state.referenceH, 1 / 480);
            assert.equal(state.pbd.selected.length, expected.pbd.selected);
            assert.equal(state.pbd.reference.length, expected.pbd.reference);
            assert.equal(state.xpbd.selected.length, expected.xpbd.selected);
            assert.equal(state.xpbd.reference.length, expected.xpbd.reference);
            assert.equal(state.lengthDifference.pbd,
                Math.abs(state.pbd.selected.length - state.pbd.reference.length));
            assert.equal(state.lengthDifference.xpbd,
                Math.abs(state.xpbd.selected.length - state.xpbd.reference.length));
            for (const method of ['pbd', 'xpbd']) {
                assert.equal(state[method].selected.error,
                    Math.abs(state[method].selected.length - 2) / 2);
                assert.ok(state[method].selected.maxRelativeError >= state[method].selected.error);
            }
            const text = readout(root).textContent;
            for (const marker of ['PBD 所选最终长度', 'PBD 参考最终长度', 'XPBD 所选最终长度',
                'XPBD 参考最终长度', '所选与参考绝对长度差', '相对约束误差', '总模拟时间',
                '参考子步数', '所选子步数']) assert.ok(text.includes(marker), marker);
            assert.doesNotMatch(text, /NaN|Infinity/);
            const verifiedCalls = {realCalls, lambdaLifecycle, pbdCalls, xpbdCalls};
            const controls = inputs(root);
            controls[0].value = '1'; controls[1].value = '3'; controls[2].value = '0.02';
            controls[0].dispatch('input');
            state = demo.getState();
            assert.equal(state.frame, 0);
            assert.equal(state.time, 0);
            assert.equal(state.selectedSubsteps, 1);
            assert.equal(state.iterations, 3);
            assert.equal(state.compliance, 0.02);
            assert.equal(state.xpbd.selected.lambda, 0);
            demo.step();
            assert.equal(xpbdCalls - 80, 27, 'two XPBD runs: (1+8)*3');
            console.log(JSON.stringify({realCalls: verifiedCalls.realCalls,
                lambdaLifecycle: verifiedCalls.lambdaLifecycle,
                pbdCalls: verifiedCalls.pbdCalls, xpbdCalls: verifiedCalls.xpbdCalls,
                differences: state.lengthDifference}));
        """, setup="""
            const shared = window.RTPhysics;
            const pbdFn = shared.projectDistance, xpbdFn = shared.projectDistanceXPBD;
            let pbdCalls = 0, xpbdCalls = 0, realCalls = true, lambdaLifecycle = true;
            let expectedLambda = 0, nextLambdaReset = false;
            function predict(pair, h) {
                return pair.map(p => Object.assign({}, p, {
                    x: p.x + p.vx * h, y: p.y + p.vy * h,
                }));
            }
            function solve(method, substeps, iterations, compliance) {
                let pair = [{x: -1.5, y: 0, vx: -0.6, vy: 0, w: 1},
                    {x: 1.5, y: 0, vx: 0.6, vy: 0, w: 1}];
                const h = 1 / (60 * substeps);
                for (let step = 0; step < substeps; step += 1) {
                    const old = pair;
                    pair = predict(pair, h);
                    let lambda = 0;
                    for (let iteration = 0; iteration < iterations; iteration += 1) {
                        if (method === 'pbd') pair = Object.values(pbdFn(pair[0], pair[1], 2, 1)).slice(0, 2);
                        else {
                            const result = xpbdFn(pair[0], pair[1], 2, compliance, h, lambda);
                            pair = [result.a, result.b]; lambda = result.lambda;
                        }
                    }
                    pair = pair.map((p, i) => Object.assign({}, p, {
                        vx: (p.x - old[i].x) / h, vy: (p.y - old[i].y) / h,
                    }));
                }
                return Math.hypot(pair[1].x - pair[0].x, pair[1].y - pair[0].y);
            }
            const expected = {
                pbd: {selected: solve('pbd', 2, 8, 0.01), reference: solve('pbd', 8, 8, 0.01)},
                xpbd: {selected: solve('xpbd', 2, 8, 0.01), reference: solve('xpbd', 8, 8, 0.01)},
            };
            shared.projectDistance = (...args) => { pbdCalls += 1; return pbdFn(...args); };
            shared.projectDistanceXPBD = (...args) => {
                xpbdCalls += 1;
                if (![1 / 120, 1 / 480].includes(args[4]) || args[2] !== 2 || args[3] !== 0.01) {
                    realCalls = false;
                }
                if (nextLambdaReset) {
                    if (args[5] !== 0) lambdaLifecycle = false;
                    nextLambdaReset = false;
                }
                if (args[5] !== expectedLambda) lambdaLifecycle = false;
                const result = xpbdFn(...args);
                expectedLambda = result.lambda;
                xpbdCallsSinceReset = (xpbdCallsSinceReset + 1) % 8;
                if (xpbdCallsSinceReset === 0) { expectedLambda = 0; nextLambdaReset = true; }
                return result;
            };
            let xpbdCallsSinceReset = 0;
        """)
        self.assertTrue(data["realCalls"])
        self.assertTrue(data["lambdaLifecycle"])
        self.assertEqual((data["pbdCalls"], data["xpbdCalls"]), (80, 80))

    def test_substep_demo_bounds_playback_draws_common_error_bars_and_recovers_from_invalid_values(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics16.js"], """
            const assert = require('node:assert/strict');
            const demo = mounts[0];
            const root = document.getElementById('substep-invariance');
            demo.canvas.clientWidth = 328; demo.canvas.clientHeight = 180;
            demo.canvas.textDraws = []; demo.canvas.pathDraws = [];
            demo.draw();
            assert.equal(demo.canvas.width, 328);
            assert.equal(demo.canvas.height, 180);
            for (const call of demo.canvas.pathDraws) {
                assert.ok(Number.isFinite(call.x) && Number.isFinite(call.y));
                assert.ok(call.x >= 0 && call.x <= 328);
                assert.ok(call.y >= 0 && call.y <= 180);
            }
            for (const call of demo.canvas.textDraws) {
                assert.ok(call.x >= 0 && call.x + call.text.length * 13 <= 328, call.text);
                assert.ok(call.y >= 0 && call.y <= 180, call.text);
            }
            assert.ok(demo.canvas.textDraws.some(call => call.text.includes('误差条')));
            demo.setRunning(true);
            for (let frame = 0; frame < 60; frame += 1) demo.step();
            let state = demo.getState();
            assert.equal(state.frame, 60);
            assert.equal(state.finished, true);
            assert.equal(state.finite, true);
            assert.ok(Math.abs(state.time - 1) < 1e-12);
            assert.equal(pending.size, 0);
            const stopped = JSON.parse(JSON.stringify(state));
            demo.step();
            assert.deepEqual(JSON.parse(JSON.stringify(demo.getState())), stopped);
            demo.reset(); demo.step();
            const first = JSON.parse(JSON.stringify(demo.getState()));
            demo.reset(); demo.step();
            assert.deepEqual(JSON.parse(JSON.stringify(demo.getState())), first);
            demo.reset();
            corrupt = true; demo.step();
            state = demo.getState();
            assert.equal(state.invalid, true);
            assert.equal(state.finite, false);
            assert.equal(pending.size, 0);
            const warning = children(root).find(element => element.attributes.role === 'status');
            assert.match(warning.textContent, /非有限/);
            assert.match(warning.textContent, /重置/);
            assert.doesNotMatch(readout(root).textContent, /NaN|Infinity/);
            demo.reset();
            const controls = inputs(root);
            controls[0].value = String(Number.NaN); controls[0].dispatch('input');
            assert.equal(demo.getState().invalid, true);
            assert.equal(demo.getState().finite, false);
            assert.equal(demo.getState().selectedSubsteps, 2);
            assert.equal(pending.size, 0);
            demo.reset();
            assert.deepEqual(inputs(root).map(input => Number(input.value)), [2, 8, 0.01]);
            assert.equal(demo.getState().invalid, false);
            assert.equal(warning.hidden, true);
            console.log(JSON.stringify({fits: true, finished: true, deterministic: true,
                invalidPaused: true, invalidControlPaused: true}));
        """, setup="""
            let corrupt = false;
            const xpbdFn = window.RTPhysics.projectDistanceXPBD;
            window.RTPhysics.projectDistanceXPBD = (...args) => {
                const result = xpbdFn(...args);
                if (corrupt) { corrupt = false; result.b.x = Number.NaN; }
                return result;
            };
        """)
        self.assertEqual(data, {"fits": True, "finished": True, "deterministic": True,
                                "invalidPaused": True, "invalidControlPaused": True})

    def test_chapter_eight_demos_are_independent_and_reduced_motion_safe(self):
        data = self.run_demos([
            REPO_DIR / "book/assets/demos/physics15.js",
            REPO_DIR / "book/assets/demos/physics16.js",
        ], """
            const assert = require('node:assert/strict');
            const race = mounts[0], invariance = mounts[1];
            assert.equal(mounts.length, 2);
            assert.notEqual(race.canvas, invariance.canvas);
            assert.equal(pending.size, 0);
            race.step();
            assert.equal(invariance.getState().frame, 0);
            invariance.step();
            assert.equal(race.getState().frame, 1);
            race.setRunning(true); invariance.setRunning(true);
            assert.equal(pending.size, 2);
            race.reset();
            assert.equal(pending.size, 1);
            invariance.reset();
            assert.equal(pending.size, 0);
            assert.match(readout(document.getElementById('solver-race')).textContent, /动能变化/);
            assert.match(readout(document.getElementById('substep-invariance')).textContent, /长度差/);
            console.log(JSON.stringify({independent: true, reducedMotion: true}));
        """, setup="""
            window.matchMedia = () => ({matches: true});
        """)
        self.assertEqual(data, {"independent": True, "reducedMotion": True})


class PhysicsChapterNineDemoTests(unittest.TestCase):
    run_demos = PhysicsChapterDemoTests.run_demos

    def test_rigid_contact_uses_real_shared_response_and_approaching_contact_episodes(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics17.js"], """
            const assert = require('node:assert/strict');
            const demo = mounts[0];
            const root = document.getElementById('rigid-contact');
            assert.deepEqual(inputs(root).map(input => Number(input.value)), [0.35, 0.58, 0.18, 1.8]);
            assert.equal(pending.size, 0);
            const initial = demo.getState();
            assert.deepEqual(JSON.parse(JSON.stringify(initial.particle)),
                {x: 0, y: 2.35, vx: 1.8, vy: 0, w: 1});
            assert.equal(initial.radius, 0.35);
            assert.equal(initial.contacts, 0);
            assert.equal(initial.gravity, -9.81);
            assert.equal(initial.totalDuration, 6);
            assert.equal(initial.substepDt, 1 / 120);
            assert.match(demo.canvas.attributes['aria-label'], /平移.*地面.*接触/);
            Object.assign(demo.getState(), {radius: 0.5, restitution: 0.4, friction: 0.25});
            Object.assign(demo.getState().particle, {x: 0, y: 0.501, vx: 2, vy: -1, w: 1});
            const before = JSON.parse(JSON.stringify(demo.getState().particle));
            demo.step();
            const state = demo.getState();
            assert.equal(colliderCalls, 2);
            const firstColliderCalls = colliderCalls;
            assert.ok(realCalls && unchangedInputs);
            assert.deepEqual(before, {x: 0, y: 0.501, vx: 2, vy: -1, w: 1});
            const event = contactEvents[0];
            assert.equal(event.before.x, 2 / 120);
            assert.ok(Math.abs(event.before.y - 0.4919854166666667) < 1e-12);
            assert.equal(event.before.vx, 2);
            assert.equal(event.before.vy, -1.08175);
            assert.equal(event.before.w, 1);
            assert.equal(event.after.x, 2 / 120);
            assert.equal(event.after.y, 0.5);
            assert.equal(event.after.vx, 1.5);
            assert.ok(Math.abs(event.after.vy - 0.4327) < 1e-12);
            assert.equal(event.after.w, 1);
            assert.equal(state.contacts, 1);
            assert.ok(Math.abs(state.particle.y - (0.5 + 0.35095 / 120)) < 1e-12);
            assert.equal(state.particle.vx, 1.5);
            assert.ok(Math.abs(state.particle.vy - 0.35095) < 1e-12);
            assert.equal(state.slidingDistance, 1.5 / 120,
                'only later response-to-response travel counts after the first contact response');
            assert.equal(state.bounceHeight, state.particle.y - 0.5);

            demo.reset();
            Object.assign(demo.getState(), {radius: 0.5, restitution: 0.1, friction: 0.25});
            Object.assign(demo.getState().particle, {x: 0, y: 0.5, vx: 2, vy: 0, w: 1});
            for (let frame = 0; frame < 60; frame += 1) demo.step();
            assert.equal(demo.getState().contacts, 1,
                'substep micro-bounces that never leave the floor remain one contact episode');
            assert.equal(demo.getState().particle.y, 0.5);
            assert.equal(demo.getState().bounceHeight, 0);
            assert.ok(demo.getState().slidingDistance > 0);
            console.log(JSON.stringify({realCalls, unchangedInputs, colliderCalls: firstColliderCalls,
                firstContact: contactEvents[0], restingContacts: 1}));
        """, setup="""
            const collide = window.RTPhysics.collideSpherePlane;
            const contactEvents = [];
            let colliderCalls = 0, realCalls = true, unchangedInputs = true;
            window.RTPhysics.collideSpherePlane = (particle, radius, planeY, restitution, friction) => {
                colliderCalls += 1;
                if (planeY !== 0 || particle.w !== 1 || restitution < 0 || restitution > 1 ||
                    friction < 0 || friction > 1 || !(radius > 0)) realCalls = false;
                const before = JSON.parse(JSON.stringify(particle));
                const result = collide(particle, radius, planeY, restitution, friction);
                if (JSON.stringify(particle) !== JSON.stringify(before) || result === particle) unchangedInputs = false;
                if (particle.vy < 0 && particle.y <= planeY + radius) {
                    contactEvents.push({before, after: JSON.parse(JSON.stringify(result))});
                }
                return result;
            };
        """)
        self.assertTrue(data["realCalls"])
        self.assertTrue(data["unchangedInputs"])
        self.assertEqual(data["colliderCalls"], 2)
        self.assertEqual(data["firstContact"]["after"]["x"], 2 / 120)
        self.assertEqual(data["firstContact"]["after"]["y"], 0.5)
        self.assertEqual(data["firstContact"]["after"]["vx"], 1.5)
        self.assertAlmostEqual(data["firstContact"]["after"]["vy"], 0.4327)
        self.assertEqual(data["firstContact"]["after"]["w"], 1)

    def test_rigid_contact_measures_post_contact_bounce_slide_and_finite_extremes(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics17.js"], """
            const assert = require('node:assert/strict');
            const demo = mounts[0];
            const root = document.getElementById('rigid-contact');
            const controls = inputs(root);
            const cases = [[0.35, 0.58, 0.18, 1.8], [0.2, 0, 1, 0],
                [0.8, 1, 0, -3], [0.8, 0, 0, 3], [0.2, 1, 1, -3]];
            for (const values of cases) {
                values.forEach((value, index) => { controls[index].value = String(value); });
                controls[0].dispatch('input');
                for (let frame = 0; frame < 360; frame += 1) {
                    demo.step();
                    const state = demo.getState();
                    assert.equal(state.finite, true);
                    assert.equal(state.frame, frame + 1);
                    assert.ok(Math.abs(state.time - (frame + 1) / 60) < 1e-12);
                    for (const value of [state.particle.x, state.particle.y, state.particle.vx,
                        state.particle.vy, state.contacts, state.bounceHeight, state.slidingDistance]) {
                        assert.ok(Number.isFinite(value), String(value));
                    }
                    assert.equal(state.bounceHeight, expectedMaxCenter === null ? 0 : expectedMaxCenter - state.radius);
                    assert.ok(Math.abs(state.slidingDistance - expectedSlide) < 1e-10,
                        'slide ' + state.slidingDistance + ' expected ' + expectedSlide + ' frame ' + frame +
                        ' max ' + expectedMaxCenter + ' contacts ' + state.contacts);
                }
                const state = demo.getState();
                assert.equal(state.finished, true);
                assert.equal(state.contacts > 0, true);
                assert.equal(state.radius, values[0]);
                assert.equal(state.restitution, values[1]);
                assert.equal(state.friction, values[2]);
                assert.equal(state.initialVx, values[3]);
                assert.equal(state.particle.w, 1);
                assert.equal(state.gravity, -9.81);
                assert.doesNotMatch(readout(root).textContent, /NaN|Infinity/);
            }
            demo.reset();
            const defaultBounce = demo.getState();
            for (let frame = 0; frame < 360; frame += 1) demo.step();
            const final = demo.getState();
            assert.ok(final.bounceHeight > 0.5 && final.bounceHeight < 0.8);
            assert.ok(final.slidingDistance > 1, 'default slide=' + final.slidingDistance +
                ' bounce=' + final.bounceHeight + ' contacts=' + final.contacts);
            assert.ok(final.contacts >= 3);
            assert.ok(final.contacts < 30, 'resting micro-bounces must not inflate contact episodes');
            demo.reset();
            assert.deepEqual(inputs(root).map(input => Number(input.value)), [0.35, 0.58, 0.18, 1.8]);
            assert.equal(demo.getState().particle.y, 2.35);
            assert.equal(demo.getState().contacts, 0);
            demo.step();
            const first = JSON.parse(JSON.stringify(demo.getState()));
            demo.reset(); demo.step();
            assert.deepEqual(JSON.parse(JSON.stringify(demo.getState())), first);
            const text = readout(root).textContent;
            for (const marker of ['接触次数', '反弹高度', '滑行距离', '当前 vx', '当前 vy',
                                  '物理时间', '有限值', '接触高度']) assert.ok(text.includes(marker), marker);
            console.log(JSON.stringify({defaultBounce: final.bounceHeight, defaultSlide: final.slidingDistance,
                contacts: final.contacts, deterministic: true, finite: true}));
        """, setup="""
            const collide = window.RTPhysics.collideSpherePlane;
            let expectedMaxCenter = null, expectedSlide = 0, previousResponseX = null;
            document.getElementById('rigid-contact').addEventListener('input', () => {
                expectedMaxCenter = null; expectedSlide = 0; previousResponseX = null;
            });
            window.RTPhysics.collideSpherePlane = (...args) => {
                const result = collide(...args);
                if (args[0].vy < 0 && args[0].y <= args[1] && expectedMaxCenter === null) {
                    expectedMaxCenter = result.y;
                    previousResponseX = result.x;
                } else if (expectedMaxCenter !== null) {
                    expectedMaxCenter = Math.max(expectedMaxCenter, result.y);
                    expectedSlide += Math.abs(result.x - previousResponseX);
                    previousResponseX = result.x;
                }
                return result;
            };
        """)
        self.assertTrue(data["deterministic"])
        self.assertTrue(data["finite"])
        self.assertGreater(data["defaultBounce"], 0.5)
        self.assertGreater(data["defaultSlide"], 1)

    def test_rigid_contact_narrow_geometry_finishes_cancels_raf_and_recovers_without_clipping(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics17.js"], """
            const assert = require('node:assert/strict');
            const demo = mounts[0];
            const root = document.getElementById('rigid-contact');
            demo.canvas.clientWidth = 328; demo.canvas.clientHeight = 180;
            let checkedDraws = 0, sawContactGeometry = false;
            function checkDraw() {
                demo.canvas.pathDraws = []; demo.canvas.textDraws = [];
                demo.draw();
                assert.equal(demo.canvas.width, 328);
                assert.equal(demo.canvas.height, 180);
                for (const call of demo.canvas.pathDraws) {
                    assert.ok(Number.isFinite(call.x) && Number.isFinite(call.y));
                    assert.ok(call.x >= 0 && call.x <= 328);
                    assert.ok(call.y >= 0 && call.y <= 180);
                }
                for (const call of demo.canvas.textDraws) {
                    assert.ok(Number.isFinite(call.x) && Number.isFinite(call.y));
                    assert.ok(call.x >= 0 && call.x + call.text.length * 13 <= 328, call.text);
                    assert.ok(call.y >= 0 && call.y <= 180, call.text);
                }
                const ball = demo.canvas.pathDraws.find(call => call.op === 'arc');
                assert.ok(ball && ball.r > 0);
                assert.ok(ball.x - ball.r >= 0 && ball.x + ball.r <= 328);
                assert.ok(ball.y - ball.r >= 0 && ball.y + ball.r <= 180);
                const floor = demo.canvas.pathDraws.find(call => call.op === 'moveTo' && call.y === 150);
                assert.ok(floor, 'ground line at y=150');
                if (demo.getState().contacts > 0 && demo.getState().particle.y === demo.getState().radius) {
                    assert.ok(Math.abs(ball.y + ball.r - floor.y) < 1e-9);
                    sawContactGeometry = true;
                }
                checkedDraws += 1;
            }
            checkDraw();
            demo.setRunning(true);
            assert.equal(pending.size, 1);
            let rafTicks = 0;
            while (!demo.getState().finished && rafTicks < 400) {
                rafTicks += 1;
                tick(rafTicks * 1000 / 60);
                if (demo.getState().contacts > 0 && !sawContactGeometry) {
                    const actualY = demo.getState().particle.y;
                    demo.getState().particle.y = demo.getState().radius;
                    checkDraw();
                    demo.getState().particle.y = actualY;
                }
                checkDraw();
            }
            let state = demo.getState();
            assert.equal(state.finished, true);
            assert.equal(state.frame, 360);
            assert.equal(state.finite, true);
            assert.equal(pending.size, 0);
            assert.ok(sawContactGeometry, 'contact ball bottom must meet the drawn floor');
            const stopped = JSON.parse(JSON.stringify(state));
            demo.step();
            assert.deepEqual(JSON.parse(JSON.stringify(demo.getState())), stopped);
            demo.setRunning(true);
            tick((rafTicks + 1) * 1000 / 60);
            tick((rafTicks + 1) * 1000 / 60 + 100);
            assert.equal(pending.size, 0, 'finished restart retains no callback');
            demo.setRunning(true);
            assert.equal(pending.size, 1);
            const retained = requested[requested.length - 1];
            demo.reset();
            assert.equal(pending.size, 0);
            assert.ok(cancelled.includes(retained), 'reset must cancel its retained RAF');
            demo.reset();
            assert.equal(state.frame, 360);
            demo.setRunning(true); assert.equal(pending.size, 1);
            demo.reset(); assert.equal(pending.size, 0);
            demo.step();
            const first = JSON.parse(JSON.stringify(demo.getState()));
            demo.reset(); demo.step();
            assert.deepEqual(JSON.parse(JSON.stringify(demo.getState())), first);
            demo.reset();
            corrupt = true; demo.step();
            state = demo.getState();
            assert.equal(state.finite, false);
            assert.equal(state.invalid, true);
            assert.ok(Number.isNaN(state.particle.y), 'numerical errors remain visible in state');
            assert.doesNotMatch(readout(root).textContent, /NaN|Infinity/);
            const warning = children(root).find(element => element.attributes.role === 'status');
            assert.equal(warning.hidden, false);
            assert.match(warning.textContent, /非有限/);
            assert.match(warning.textContent, /重置/);
            demo.reset();
            assert.equal(demo.getState().finite, true);
            assert.equal(warning.hidden, true, 'reset clears contact warning');
            console.log(JSON.stringify({fits: true, checkedDraws, sawContactGeometry,
                finished: true, cancelledRestart: true, deterministic: true, invalidPaused: true}));
        """, setup="""
            let corrupt = false;
            const collide = window.RTPhysics.collideSpherePlane;
            window.RTPhysics.collideSpherePlane = (...args) => {
                const result = collide(...args);
                if (corrupt) { corrupt = false; result.y = Number.NaN; }
                return result;
            };
        """)
        self.assertTrue(data["fits"])
        self.assertTrue(data["sawContactGeometry"])
        self.assertTrue(data["cancelledRestart"])
        self.assertTrue(data["invalidPaused"])

    def test_rotation_intuition_reconstructs_no_slip_formulas_and_finite_time_horizons(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics18.js"], """
            const assert = require('node:assert/strict');
            const demo = mounts[0];
            const root = document.getElementById('rotation-intuition');
            assert.deepEqual(inputs(root).map(input => Number(input.value)), [1.2, 0.125, 2]);
            assert.equal(pending.size, 0);
            let state = demo.getState();
            assert.equal(state.mass, 1);
            assert.equal(state.radius, 0.5);
            assert.equal(state.inertia, 0.125);
            assert.equal(state.torque, 1.2);
            assert.equal(state.horizon, 2);
            assert.equal(state.duration, 120);
            assert.ok(Math.abs(state.alpha - 3.2) < 1e-12);
            assert.equal(state.omega, 0);
            assert.equal(state.theta, 0);
            assert.equal(state.x, 0);
            assert.match(demo.canvas.attributes['aria-label'], /理想无滑动.*圆盘.*径向标记/);
            demo.step();
            state = demo.getState();
            assert.ok(Math.abs(state.time - 1 / 60) < 1e-12);
            assert.ok(Math.abs(state.omega - 3.2 / 60) < 1e-12);
            assert.ok(Math.abs(state.theta - 0.5 * 3.2 / 3600) < 1e-12);
            assert.equal(state.rimArc, state.radius * state.theta);
            assert.equal(state.x, state.radius * state.theta);
            assert.equal(state.v, state.radius * state.omega);
            const controls = inputs(root);
            const cases = [[1.2, 0.125, 2], [4, 0.125, 3], [0, 0.25, 0.5],
                [2, 0.25, 1], [4, 0.25, 0.5]];
            for (const values of cases) {
                values.forEach((value, index) => { controls[index].value = String(value); });
                controls[0].dispatch('input');
                state = demo.getState();
                const duration = Math.round(values[2] * 60);
                assert.equal(state.frame, 0);
                assert.equal(state.time, 0);
                assert.equal(state.torque, values[0]);
                assert.equal(state.inertia, values[1]);
                assert.equal(state.horizon, values[2]);
                assert.equal(state.duration, duration);
                assert.equal(state.mass, 1);
                assert.equal(state.radius, 0.5);
                const alpha = values[0] / (values[1] + 0.25);
                assert.ok(Math.abs(state.alpha - alpha) < 1e-12);
                for (let frame = 0; frame < duration; frame += 1) {
                    demo.step();
                    state = demo.getState();
                    const time = (frame + 1) / 60;
                    assert.equal(state.finite, true);
                    assert.ok(Math.abs(state.omega - alpha * state.time) < 1e-12);
                    assert.ok(Math.abs(state.theta - 0.5 * alpha * state.time ** 2) < 1e-12);
                    assert.equal(state.rimArc, 0.5 * state.theta);
                    assert.equal(state.x, 0.5 * state.theta);
                    assert.ok(Math.abs(state.time - time) < 1e-12);
                }
                assert.equal(state.finished, true);
                assert.equal(state.time, values[2]);
                assert.ok(Math.abs(state.omega - alpha * values[2]) < 1e-12);
                assert.ok(Math.abs(state.theta - 0.5 * alpha * values[2] ** 2) < 1e-12);
                const text = readout(root).textContent;
                for (const marker of ['角加速度', '角速度', '转角', '弧长', '平移', '物理时间',
                                      '有限值', '实验时间范围', '有效惯量']) assert.ok(text.includes(marker), marker);
                assert.doesNotMatch(text, /NaN|Infinity/);
            }
            demo.reset();
            assert.deepEqual(inputs(root).map(input => Number(input.value)), [1.2, 0.125, 2]);
            demo.step();
            const first = JSON.parse(JSON.stringify(demo.getState()));
            demo.reset(); demo.step();
            assert.deepEqual(JSON.parse(JSON.stringify(demo.getState())), first);
            console.log(JSON.stringify({defaultFinal: {alpha: 3.2, omega: 6.4, theta: 6.4,
                rimArc: 3.2, x: 3.2}, deterministic: true, finite: true}));
        """)
        self.assertTrue(data["deterministic"])
        self.assertTrue(data["finite"])

    def test_rotation_intuition_draws_marker_ground_arc_and_finishes_without_retained_raf(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics18.js"], """
            const assert = require('node:assert/strict');
            const demo = mounts[0];
            const root = document.getElementById('rotation-intuition');
            demo.canvas.clientWidth = 328; demo.canvas.clientHeight = 180;
            let checkedDraws = 0;
            function checkDraw() {
                demo.canvas.pathDraws = []; demo.canvas.textDraws = [];
                demo.draw();
                assert.equal(demo.canvas.width, 328);
                assert.equal(demo.canvas.height, 180);
                const state = demo.getState();
                for (const call of demo.canvas.pathDraws) {
                    assert.ok(Number.isFinite(call.x) && Number.isFinite(call.y));
                    assert.ok(call.x >= 0 && call.x <= 328);
                    assert.ok(call.y >= 0 && call.y <= 180);
                }
                for (const call of demo.canvas.textDraws) {
                    assert.ok(Number.isFinite(call.x) && Number.isFinite(call.y));
                    assert.ok(call.x >= 0 && call.x + call.text.length * 13 <= 328, call.text);
                    assert.ok(call.y >= 0 && call.y <= 180, call.text);
                }
                const disk = demo.canvas.pathDraws.find(call => call.op === 'arc' && call.r > 0);
                const indicator = demo.canvas.pathDraws.find(call => call.op === 'arc' && call.r > disk.r);
                assert.ok(disk, 'disk circumference');
                assert.ok(indicator, 'angular arc indicator');
                assert.equal(indicator.start, 0);
                assert.ok(Math.abs(indicator.end - state.theta % (2 * Math.PI)) < 1e-12);
                assert.ok(demo.canvas.pathDraws.some(call => call.op === 'moveTo' && call.y === 150),
                    'ground line');
                const angle = state.theta % (2 * Math.PI);
                const markerX = disk.x + disk.r * Math.cos(angle);
                const markerY = disk.y + disk.r * Math.sin(angle);
                const markerStart = demo.canvas.pathDraws.findIndex(call => call.op === 'moveTo' &&
                    Math.abs(call.x - disk.x) < 1e-12 && Math.abs(call.y - disk.y) < 1e-12);
                assert.ok(markerStart >= 0, 'radial marker starts at the disk center');
                const markerEnd = demo.canvas.pathDraws[markerStart + 1];
                assert.ok(markerEnd && markerEnd.op === 'lineTo', 'disk center belongs to a radial marker');
                assert.ok(Math.abs(markerEnd.x - markerX) < 1e-12 &&
                    Math.abs(markerEnd.y - markerY) < 1e-12,
                    'radial marker endpoint follows theta: expected ' + markerX + ',' + markerY +
                    ' got ' + markerEnd.x + ',' + markerEnd.y);
                const markerDot = demo.canvas.pathDraws.find(call => call.op === 'arc' &&
                    Math.abs(call.x - markerX) < 1e-12 && Math.abs(call.y - markerY) < 1e-12 &&
                    Math.abs(call.r - Math.max(1.5, disk.r * 0.1)) < 1e-12);
                assert.ok(markerDot, 'marker dot shares the theta-derived endpoint and expected radius');
                assert.ok(demo.canvas.textDraws.some(call => call.text.includes('α=')));
                assert.ok(demo.canvas.textDraws.some(call => call.text.includes('ω=')));
                assert.ok(demo.canvas.textDraws.some(call => call.text.includes('θ=')));
                checkedDraws += 1;
            }
            checkDraw();
            for (let frame = 1; frame <= 120; frame += 1) {
                demo.step();
                if (frame % 20 === 0 || frame === 120) checkDraw();
            }
            let state = demo.getState();
            assert.equal(state.finished, true);
            assert.equal(state.frame, 120);
            assert.equal(state.time, 2);
            assert.ok(Math.abs(state.theta - 6.4) < 1e-12);
            assert.ok(Math.abs(state.x - 3.2) < 1e-12);
            const stopped = JSON.parse(JSON.stringify(state));
            demo.step();
            assert.deepEqual(JSON.parse(JSON.stringify(demo.getState())), stopped);
            demo.setRunning(true);
            tick(0); tick(1000 / 60);
            assert.equal(pending.size, 0, 'finished restart retains no callback');
            demo.setRunning(true); assert.equal(pending.size, 1);
            const retained = requested[requested.length - 1];
            demo.reset();
            assert.ok(cancelled.includes(retained));
            demo.reset(); demo.setRunning(true); assert.equal(pending.size, 1);
            demo.reset(); assert.equal(pending.size, 0);
            const controls = inputs(root);
            controls[0].value = String(Number.NaN); controls[0].dispatch('input');
            state = demo.getState();
            assert.equal(state.invalid, true);
            assert.equal(state.finite, false);
            assert.equal(state.torque, 1.2);
            assert.doesNotMatch(readout(root).textContent, /NaN|Infinity/);
            const warning = children(root).find(element => element.attributes.role === 'status');
            assert.match(warning.textContent, /非有限/);
            demo.reset();
            assert.equal(warning.hidden, true);
            console.log(JSON.stringify({fits: true, checkedDraws, marker: true, ground: true,
                arc: true, finished: true, cancelledRestart: true, invalidPaused: true}));
        """)
        self.assertTrue(data["fits"])
        self.assertTrue(data["cancelledRestart"])
        self.assertTrue(data["invalidPaused"])

    def test_chapter_nine_demos_are_independent_and_reduced_motion_allows_single_step(self):
        data = self.run_demos([
            REPO_DIR / "book/assets/demos/physics17.js",
            REPO_DIR / "book/assets/demos/physics18.js",
        ], """
            const assert = require('node:assert/strict');
            const contact = mounts[0], rotation = mounts[1];
            assert.equal(mounts.length, 2);
            assert.notEqual(contact.canvas, rotation.canvas);
            assert.equal(pending.size, 0);
            assert.equal(document.getElementById('rigid-contact').children.find(
                element => element.className === 'demo-toolbar physics-controls').children[1].textContent, '运行');
            contact.step();
            assert.equal(rotation.getState().frame, 0);
            rotation.step();
            assert.equal(contact.getState().frame, 1);
            assert.equal(rotation.getState().frame, 1);
            contact.reset();
            assert.equal(contact.getState().frame, 0);
            assert.equal(rotation.getState().frame, 1);
            rotation.reset();
            assert.equal(rotation.getState().frame, 0);
            contact.setRunning(true); rotation.setRunning(true);
            assert.equal(pending.size, 2);
            contact.reset();
            assert.equal(pending.size, 1);
            rotation.reset();
            assert.equal(pending.size, 0);
            console.log(JSON.stringify({independent: true, reducedMotion: true, singleStep: true}));
        """, setup="""
            window.matchMedia = () => ({matches: true});
        """)
        self.assertEqual(data, {"independent": True, "reducedMotion": True, "singleStep": True})


class PhysicsChapterTenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tempdir.cleanup)
        cls.output = Path(cls.tempdir.name)
        physics_build.build(cls.output)
        cls.rows = {row.number: row for row in physics_content.chapters()}

    def test_chapter_ten_has_twelve_numbered_sections_and_real_demo_bindings(self):
        self.assertIn(10, self.rows, "chapter 10 must be available")
        row = self.rows[10]
        self.assertEqual((row.slug, row.unit, row.title),
                         ("physics10", "单元四：从点到刚体和软体", "软体、布料与综合项目"))
        self.assertEqual(len(row.sections), 12)
        self.assertEqual([section.id for section in row.sections], [
            "multiparticle-softbody", "triangular-mesh", "constraint-families", "pinned-particles",
            "strain-limit", "tetrahedral-volume", "volume-gradient", "particle-rigid-coupling",
            "self-collision", "integrated-scene", "performance-metrics", "roadmap-debugging",
        ])
        for section in row.sections:
            self.assertGreater(len(section.html), 500, section.id)
            self.assertIn("<p>", section.html)
            self.assertNotIn("$", section.html)
        self.assertEqual([demo.id for demo in row.demos], ["cloth-showcase", "soft-area"])
        self.assertEqual([demo.section_id for demo in row.demos],
                         ["integrated-scene", "volume-gradient"])
        self.assertEqual([demo.script for demo in row.demos],
                         ["demos/physics19.js", "demos/physics20.js"])
        text = (self.output / "physics/chapters/physics10.html").read_text(encoding="utf-8")
        for number in range(1, 13):
            self.assertIn(f"10.{number}", text)
        tree = HTMLTree(text)
        for demo_id, section_id in (("cloth-showcase", "integrated-scene"),
                                    ("soft-area", "volume-gradient")):
            demo = next(element for element in tree.elements if element["attrs"].get("id") == demo_id)
            parent = next(element for element in tree.elements
                          if element["attrs"].get("id") == demo["parent"]["attrs"].get("id"))
            self.assertEqual(parent["attrs"]["id"], section_id)

    def test_chapter_ten_states_actual_python_softbody_cloth_and_browser_boundaries(self):
        sections = {section.id: section.html for section in self.rows[10].sections}
        text = "".join(sections.values())
        for marker in ("软体", "布料", "应变限制", "四面体", "体积约束", "自碰撞",
                       "结构", "剪切", "固定", "三角网格", "性能", "调试"):
            self.assertIn(marker, text, marker)
        boundary = sections["current-boundary"] if "current-boundary" in sections else sections["roadmap-debugging"]
        for marker in ("只有结构/剪切距离约束", "六个四面体", "中心粒子不直接参与体积约束",
                       "没有面积约束", "没有独立弯曲/二面角约束", "没有自碰撞", "没有细化体积网格",
                       "每个完整固定步", "只调用一次", "collide_with_particles",
                       "float32", "双精度", "不承诺逐位一致"):
            self.assertIn(marker, boundary, marker)
        self.assertIn("第二邻点距离链接", sections["constraint-families"])
        self.assertIn("XPBD bend compliance", sections["constraint-families"])
        self.assertIn("省略", sections["integrated-scene"])
        self.assertIn("二维四粒子面积类比", sections["volume-gradient"])
        self.assertIn("不是三维体积求解器", sections["volume-gradient"])
        self.assertIn("浏览器概念实验", sections["integrated-scene"])
        self.assertIn(r"\(E_{\max}=\max_e\frac{|L_e-L_{0e}|}{L_{0e}}\)",
                      sections["performance-metrics"])
        self.assertIn(r"\(V=\frac16(\mathbf b-\mathbf a)\cdot[(\mathbf c-\mathbf a)\times(\mathbf d-\mathbf a)]\)",
                      sections["tetrahedral-volume"])
        self.assertIn(r"\(C=A-A_0\)", sections["volume-gradient"])

    def test_chapter_ten_includes_actionable_pseudocode_self_test_and_worked_example(self):
        sections = {section.id: section.html for section in self.rows[10].sections}
        integrated = sections["integrated-scene"]
        self.assertIn("求解器与碰撞伪代码", integrated)
        self.assertIn("<pre><code>", integrated)
        for marker in ("for substep", "outerTime = t", "sampleTime = outerTime + substep * h",
                       "analyticReference(sampleTime)", "applyForces(particles, h, sampleTime)",
                       "algorithm", "lambda = 0", "vx - sphere.vx",
                       "collideParticleSphere", "vx + sphere.vx"):
            self.assertIn(marker, integrated, marker)
        self.assertNotIn("analyticReference(t + substep * h)", integrated)

        debugging = sections["roadmap-debugging"]
        self.assertIn("可操作自测", debugging)
        self.assertIn("问题", debugging)
        self.assertIn("预期观察", debugging)
        self.assertIn("答案", debugging)
        self.assertIn("相对速度", debugging)
        self.assertIn("新接近接触数", debugging)

        metrics = sections["performance-metrics"]
        self.assertIn("数值算例", metrics)
        self.assertIn("4 × 8 × 106 = 3392", metrics)
        self.assertIn("不包括", metrics)
        for marker in ("外力", "碰撞", "绘图"):
            self.assertIn(marker, metrics, marker)
        self.assertEqual({source.path for source in self.rows[10].sources}, {
            "physics/cloth/geometry.py", "physics/softbody/geometry.py",
            "physics/softbody/xpbd.py", "app/physics_controller.py", "book/assets/physics-core.js",
        })

    def test_chapter_ten_formulas_render_with_strict_local_katex(self):
        text = (self.output / "physics/chapters/physics10.html").read_text(encoding="utf-8")
        formulas = [{"tex": html.unescape(tex), "display": True}
                    for tex in re.findall(r"\\\[(.*?)\\\]", text, re.DOTALL)]
        formulas.extend({"tex": html.unescape(tex), "display": False}
                        for tex in re.findall(r"\\\((.*?)\\\)", text, re.DOTALL))
        self.assertGreaterEqual(len(formulas), 80)
        result = subprocess.run([
            "node", "-e", textwrap.dedent("""
                const fs = require('node:fs');
                const katex = require(process.argv[1]);
                const formulas = JSON.parse(fs.readFileSync(0, 'utf8'));
                formulas.forEach((formula, index) => {
                  try { katex.renderToString(formula.tex, {displayMode: formula.display,
                    throwOnError: true, strict: 'error'}); }
                  catch (error) { console.error(`formula ${index}: ${error.message}: ${formula.tex}`);
                    process.exit(1); }
                });
                console.log(`Rendered ${formulas.length} LaTeX formulas`);
            """), str(REPO_DIR / "book/assets/vendor/katex/katex.min.js"),
        ], input=json.dumps(formulas), text=True, capture_output=True, timeout=10, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertIn(f"Rendered {len(formulas)} LaTeX formulas", result.stdout)


class PhysicsChapterTenDemoTests(unittest.TestCase):
    run_demos = PhysicsChapterDemoTests.run_demos

    def test_cloth_grid_connectivity_and_all_three_algorithms_use_real_substep_core(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics19.js"], """
            const assert = require('node:assert/strict');
            const demo = mounts[0], root = document.getElementById('cloth-showcase');
            const state = demo.getState();
            assert.deepEqual(inputs(root).map(input => Number(input.value)), [2, 8, 0.5, 1.2]);
            assert.equal(state.particles.length, 35);
            assert.equal(state.fixedCount, 7);
            assert.equal(state.freeCount, 28);
            assert.deepEqual(JSON.parse(JSON.stringify(state.particles.slice(0, 7).map(p => p.w))), [0,0,0,0,0,0,0]);
            assert.ok(state.particles.slice(7).every(p => p.w === 5));
            assert.equal(state.links.length, 106);
            assert.equal(state.links.filter(link => link.kind === 'structural').length, 58);
            assert.equal(state.links.filter(link => link.kind === 'shear').length, 48);
            assert.ok(state.links.every(link => link.kind !== 'bend'));
            assert.equal(state.triangles.length, 48);
            for (const link of state.links) {
              const a = state.particles[link.a], b = state.particles[link.b];
              assert.ok(Math.abs(Math.hypot(b.x-a.x, b.y-a.y)-link.restLength) < 1e-12);
            }
            assert.ok(Math.abs(state.links.find(link=>link.a===0&&link.b===1).restLength-0.45)<1e-12);
            assert.ok(Math.abs(state.links.find(link=>link.a===0&&link.b===7).restLength-0.42)<1e-12);
            assert.ok(state.links.some(link => link.a === 0 && link.b === 8 &&
              Math.abs(link.restLength-Math.hypot(0.45,0.42)) < 1e-12));
            assert.ok(state.triangles.some(t => t[0] === 0 && t[1] === 7 && t[2] === 1));
            assert.equal(pending.size, 0);
            const initial = JSON.parse(JSON.stringify(state.particles));
            const initialSphere = JSON.parse(JSON.stringify(state.sphere));
            demo.step();
            assert.equal(state.frame, 1);
            assert.ok(Math.abs(state.time-1/60) < 1e-12);
            const finalSampleTime = 3/240;
            const expectedSphere = {
              x: 1.45*Math.sin(2*Math.PI*finalSampleTime/6),
              y: 0.1+0.08*Math.sin(4*Math.PI*finalSampleTime/6+0.4),
              vx: 1.45*(2*Math.PI/6)*Math.cos(2*Math.PI*finalSampleTime/6),
              vy: 0.08*(4*Math.PI/6)*Math.cos(4*Math.PI*finalSampleTime/6+0.4),
            };
            for (const key of ['x','y','vx','vy']) {
              assert.ok(Math.abs(state.sphere[key]-expectedSphere[key]) < 1e-12,
                JSON.stringify({key, actual:state.sphere[key], expected:expectedSphere[key],
                  finalSampleTime, stateTime:state.time}));
            }
            assert.equal(xpbdCalls, 3392); assert.equal(pbdCalls, 0); assert.equal(springCalls, 0);
            assert.ok(realXPBD && lambdaReset && lambdaContinuous && positiveCompliance,
              JSON.stringify({realXPBD,lambdaReset,lambdaContinuous,positiveCompliance,lambdaMismatch}));
            assert.ok(state.particles.slice(0,7).every((p,index) => p.x === initial[index].x && p.y === initial[index].y));
            assert.ok(Math.abs(state.sphere.x-initialSphere.x) > 0);
            assert.ok(state.maxRelativeError >= 0 && Number.isFinite(state.maxRelativeError));
            const controls = inputs(root);
            controls[0].value = '1'; controls[0].dispatch('input');
            let rebuilt = demo.getState();
            assert.equal(rebuilt.algorithm, 'PBD'); assert.equal(rebuilt.frame, 0);
            assert.deepEqual(JSON.parse(JSON.stringify(rebuilt.particles)), initial);
            assert.deepEqual(JSON.parse(JSON.stringify(rebuilt.sphere)), initialSphere);
            demo.step();
            assert.equal(pbdCalls, 3392);
            assert.ok(pbdFull && pbdCalls === 4*8*106);
            controls[0].value = '0'; controls[0].dispatch('input');
            rebuilt = demo.getState();
            assert.equal(rebuilt.algorithm, '质量弹簧');
            assert.deepEqual(JSON.parse(JSON.stringify(rebuilt.particles)), initial);
            demo.step();
            assert.equal(springCalls, 400);
            assert.ok(springCalls === 4*100,'six fixed-fixed links need no force; 100 active links per substep');
            demo.reset();
            controls[0].value='0'; controls[0].dispatch('input');
            const forceState=demo.getState();
            Object.assign(forceState.particles[8], {x:forceState.particles[7].x+0.2,y:forceState.particles[7].y,
              vx:0,vy:0,w:5});
            forceState.links=[{a:7,b:8,restLength:0.45,kind:'structural'}];
            const forceA=forceState.particles[7], forceB=forceState.particles[8];
            const beforeA=forceA.x, beforeB=forceB.x;
            demo.step();
            assert.ok(forceA.x<beforeA,'positive b-directed force must move a left');
            assert.ok(forceB.x>beforeB);
            assert.match(readout(root).textContent, /最大约束误差/);
            assert.match(readout(root).textContent, /质量弹簧.*仅作诊断/);
            console.log(JSON.stringify({particles:35, fixed:7, free:28, links:106,
              structural:58, shear:48, triangles:48, lambdaReset, lambdaContinuous, pbdFull}));
        """, setup="""
            let xpbdCalls=0, pbdCalls=0, springCalls=0;
            let realXPBD=true, lambdaReset=true, lambdaContinuous=true, positiveCompliance=true;
            const previousLambdas=new Array(106).fill(0);
            let pbdFull=true, lambdaMismatch='';
            const oldXPBD=window.RTPhysics.projectDistanceXPBD;
            window.RTPhysics.projectDistanceXPBD=(a,b,L,c,dt,lambda)=>{
              xpbdCalls++;
              const index=(xpbdCalls-1)%106;
              const substepStart=xpbdCalls===1 || (xpbdCalls-1)%848===0;
              if(substepStart) { previousLambdas.fill(0); lambdaReset=lambdaReset&&lambda===0; }
              if(!substepStart && lambda!==previousLambdas[index]) {
                lambdaContinuous=false;
                if(!lambdaMismatch) lambdaMismatch=JSON.stringify({call:xpbdCalls,index,lambda,
                  previous:previousLambdas[index]});
              }
              if(!(c>0) || dt!==1/240) positiveCompliance=false;
              const result=oldXPBD(a,b,L,c,dt,lambda);
              previousLambdas[index]=result.lambda;
              realXPBD=realXPBD && result!==a && Number.isFinite(result.lambda);
              return result;
            };
            const oldPBD=window.RTPhysics.projectDistance;
            window.RTPhysics.projectDistance=(a,b,L,s)=>{pbdCalls++; if(s!==1) pbdFull=false;
              return oldPBD(a,b,L,s);};
            const oldSpring=window.RTPhysics.springForce;
            window.RTPhysics.springForce=(extension,rv,k,c)=>{springCalls++;
              const force=oldSpring(extension,rv,k,c);
              return force;};
        """)
        self.assertEqual(data["particles"], 35)
        self.assertEqual(data["links"], 106)
        self.assertEqual(data["triangles"], 48)
        self.assertTrue(data["lambdaReset"])
        self.assertTrue(data["lambdaContinuous"])
        self.assertTrue(data["pbdFull"])

    def test_cloth_collision_uses_actual_core_values_and_counts_only_new_approaching_penetration(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics19.js"], """
            const assert = require('node:assert/strict');
            const demo=mounts[0], root=document.getElementById('cloth-showcase');
            const cases=[];
            function analyticReference(time){
              return {x:1.45*Math.sin(2*Math.PI*time/6),
                y:0.1+0.08*Math.sin(4*Math.PI*time/6+0.4),
                vx:1.45*(2*Math.PI/6)*Math.cos(2*Math.PI*time/6),
                vy:0.08*(4*Math.PI/6)*Math.cos(4*Math.PI*time/6+0.4)};
            }
            for(const worldVx of [0,0.5]){
              demo.reset();
              const current=demo.getState(), sphere=current.sphere, combined=current.circleRadius+current.particleRadius;
              current.restitution=0.4; current.friction=0.25; current.links=[];
              current.gravity=0; current.wind=0;
              Object.assign(current.particles[21], {x:sphere.x+0.4, y:sphere.y, vx:worldVx, vy:0});
              const eventIndex=collisionEvents.length;
              demo.step();
              assert.equal(current.contacts, 1, JSON.stringify({worldVx,contacts:current.contacts}));
              assert.equal(collisionEvents.length,eventIndex+1);
              const event=collisionEvents[eventIndex];
              assert.ok(event.previousDistance>combined,'previous distance');
              assert.ok(event.beforeDistance<=combined+1e-12,'before distance');
              assert.ok(event.afterDistance<=combined+1e-12,'after distance');
              assert.ok(event.approaching && event.beforeDistance>1e-12,'approach');
              assert.equal(event.center.vx,0); assert.equal(event.center.vy,0);
              const reference=Array.from({length:1441},(_,index)=>analyticReference(index/240))
                .find(candidate=>Math.abs(candidate.x-event.center.x)<1e-12&&
                  Math.abs(candidate.y-event.center.y)<1e-12);
              assert.ok(reference,'collision center must lie on the analytic reference');
              assert.deepEqual(event.sphereVelocity,{vx:reference.vx,vy:reference.vy});
              assert.equal(event.before.vx,event.worldBefore.vx-event.sphereVelocity.vx);
              assert.equal(event.before.vy,event.worldBefore.vy-event.sphereVelocity.vy);
              const expectedNormal={x:(event.before.x-event.center.x)/event.beforeDistance,
                y:(event.before.y-event.center.y)/event.beforeDistance};
              const expectedVn=(event.worldBefore.vx-event.sphereVelocity.vx)*expectedNormal.x+
                (event.worldBefore.vy-event.sphereVelocity.vy)*expectedNormal.y;
              const tangent={x:event.before.vx-expectedVn*expectedNormal.x,
                y:event.before.vy-expectedVn*expectedNormal.y};
              const expectedWorldVx=event.sphereVelocity.vx+0.75*tangent.x-0.4*expectedVn*expectedNormal.x;
              const expectedWorldVy=event.sphereVelocity.vy+0.75*tangent.y-0.4*expectedVn*expectedNormal.y;
              assert.ok(Math.abs(event.after.x-(event.center.x+combined*expectedNormal.x))<1e-12,'x');
              assert.ok(Math.abs(event.after.y-(event.center.y+combined*expectedNormal.y))<1e-12,'y');
              assert.ok(Math.abs(current.particles[21].vx-expectedWorldVx)<1e-12,
                JSON.stringify({actual:current.particles[21].vx,expectedWorldVx,event}));
              assert.ok(Math.abs(current.particles[21].vy-expectedWorldVy)<1e-12,
                JSON.stringify({actual:current.particles[21].vy,expectedWorldVy,event}));
              assert.equal(coreCalls>0 && clonedInputs, true, JSON.stringify({coreCalls,clonedInputs}));
              cases.push({worldVx,worldBeforeVx:event.worldBefore.vx,worldAfterVx:current.particles[21].vx,
                sphereVx:event.sphereVelocity.vx,relativeApproaching:event.approaching});
              const contacts=current.contacts;
              current.links=[]; current.wind=0;
              for(let frame=0;frame<5;frame++) demo.step();
              assert.equal(current.contacts,contacts,'resting projected contact must not be recounted');
            }
            demo.reset();
            const degenerate=demo.getState(), h=degenerate.substepDt;
            degenerate.links=[]; degenerate.wind=0;
            Object.assign(degenerate.particles[21], {
              x:0, y:0.1+0.08*Math.sin(0.4), vx:0, vy:9.81*h});
            demo.step();
            const invalid=demo.getState();
            const warning=children(root).find(element=>element.attributes.role==='status');
            assert.equal(invalid.invalid,true,JSON.stringify({invalid:invalid.invalid,
              finite:invalid.finite,particle:invalid.particles[21],sphere:invalid.sphere,
              warning:warning.textContent})); assert.equal(invalid.finite,false);
            assert.equal(pending.size,0);
            assert.match(warning.textContent,/重合|退化/); assert.match(warning.textContent,/重置/);
            demo.reset();
            assert.equal(demo.getState().contacts,0);
            assert.equal(demo.getState().invalid,false);
            console.log(JSON.stringify({contactsPerCase:1, cases, degeneratePaused:true}));
        """, setup="""
            const collide=window.RTPhysics.collideParticleSphere;
            const collisionEvents=[]; let coreCalls=0, clonedInputs=true;
            window.RTPhysics.collideParticleSphere=(particle,center,R,r,e,f)=>{
              coreCalls++;
              const state=mounts[0].getState();
              const index=state.particles.findIndex(candidate=>candidate.x===particle.x&&candidate.y===particle.y);
              const worldBefore=Object.assign({},state.particles[index]);
              const before=JSON.parse(JSON.stringify(particle));
              const beforeDistance=Math.hypot(particle.x-center.x,particle.y-center.y);
              const combined=R+r;
              const normal=beforeDistance>1e-12?{x:(particle.x-center.x)/beforeDistance,
                y:(particle.y-center.y)/beforeDistance}:{x:0,y:0};
              const approaching=particle.vx*normal.x+particle.vy*normal.y<0;
              const previousDistance=state.lastDistance[index];
              const result=collide(particle,center,R,r,e,f);
              if(result===particle || particle===state.particles[index] || center===state.sphere ||
                 JSON.stringify(particle)!==JSON.stringify(before)) clonedInputs=false;
              const afterDistance=Math.hypot(result.x-center.x,result.y-center.y);
              if(previousDistance>combined && beforeDistance<=combined && approaching)
                collisionEvents.push({before,after:JSON.parse(JSON.stringify(result)),
                  previousDistance,beforeDistance,afterDistance,approaching,worldBefore,
                  center:{x:center.x,y:center.y,vx:center.vx,vy:center.vy},
                  sphereVelocity:{vx:state.sphere.vx,vy:state.sphere.vy}});
              return result;
            };
        """)
        self.assertEqual(data["contactsPerCase"], 1)
        self.assertTrue(data["degeneratePaused"])
        self.assertEqual(len(data["cases"]), 2)
        self.assertTrue(all(case["relativeApproaching"] for case in data["cases"]))
        self.assertTrue(all(case["worldAfterVx"] > case["worldBeforeVx"] for case in data["cases"]))
        self.assertLess(data["cases"][1]["worldBeforeVx"], data["cases"][1]["sphereVx"])

    def test_cloth_finite_horizon_control_reconstruction_drag_and_narrow_canvas(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics19.js"], """
            const assert=require('node:assert/strict');
            const demo=mounts[0], root=document.getElementById('cloth-showcase');
            const canvas=demo.canvas;
            canvas.clientWidth=328; canvas.clientHeight=180;
            function fit(){demo.canvas.pathDraws=[];demo.canvas.textDraws=[];demo.draw();
              for(const call of demo.canvas.pathDraws){
                assert.ok(Number.isFinite(call.x)&&Number.isFinite(call.x));
                assert.ok(call.x>=0&&call.x<=328&&call.y>=0&&call.y<=180,JSON.stringify(call));
                if(call.op==='arc') assert.ok(call.x-call.r>=0&&call.x+call.r<=328&&
                  call.y-call.r>=0&&call.y+call.r<=180,JSON.stringify(call));
              }
              for(const call of demo.canvas.textDraws)
                assert.ok(call.x>=0&&call.x+call.text.length*13<=328&&call.y>=0&&call.y<=180,call.text);
            }
            fit();
            demo.setRunning(true); assert.equal(pending.size,1);
            canvasDrag(demo,canvas,9999,-9999); assert.equal(pending.size,0);
            assert.equal(demo.getState().dragging,true);
            assert.ok(demo.getState().sphere.x>=-2&&demo.getState().sphere.x<=2,'sphere x');
            assert.ok(demo.getState().sphere.y>=0&&demo.getState().sphere.y<=2.8,'sphere y');
            assert.equal(demo.getState().sphere.vx,0); assert.equal(demo.getState().sphere.vy,0);
            demo.reset();
            demo.step(); const first=JSON.parse(JSON.stringify(demo.getState()));
            demo.reset(); demo.step(); assert.deepEqual(JSON.parse(JSON.stringify(demo.getState())),first);
            const controls=inputs(root);
            controls[0].value='2'; controls[1].value='40'; controls[2].value='1';
            controls[3].value='0'; controls[0].dispatch('input');
            assert.equal(demo.getState().frame,0); assert.equal(demo.getState().iterations,40);
            assert.equal(demo.getState().strength,1); assert.equal(demo.getState().wind,0);
            demo.setRunning(true);
            for(let frame=0;frame<=360;frame++) {
              assert.equal(tick(frame*1000/60+0.001),true,'RAF callback '+frame);
            }
            let state=demo.getState();
            assert.equal(state.frame,360,JSON.stringify({frame:state.frame,time:state.time,
              invalid:state.invalid,finished:state.finished}));
            assert.equal(state.finished,true);
            assert.equal(state.finite,true,JSON.stringify({frame:state.frame,time:state.time,
              invalid:state.invalid,sphere:state.sphere,particle:state.particles[21],
              warning:children(root).find(e=>e.attributes.role==='status').textContent}));
            assert.ok(Math.abs(state.time-6)<1e-12,String(state.time));
            assert.equal(pending.size,0);
            assert.equal(tick(6100),false,'finished controller must not schedule another RAF');
            const stopped=JSON.parse(JSON.stringify(state)); demo.step();
            assert.deepEqual(JSON.parse(JSON.stringify(demo.getState())),stopped);
            fit();
            demo.reset(); corrupt=true; demo.step(); state=demo.getState();
            assert.equal(state.invalid,true); assert.equal(state.finite,false); assert.equal(pending.size,0);
            assert.doesNotMatch(readout(root).textContent,/NaN|Infinity/);
            demo.reset(); controls[0].value=String(Number.NaN); controls[0].dispatch('input');
            assert.equal(demo.getState().invalid,true); assert.equal(pending.size,0);
            demo.reset(); assert.deepEqual(inputs(root).map(input=>Number(input.value)),[2,8,0.5,1.2]);
            console.log(JSON.stringify({frames:360,finite:true,deterministic:true,fits:true,
              dragPaused:true,degeneratePaused:true}));
        """, setup="""
            let corrupt=false;
            const xpbd=window.RTPhysics.projectDistanceXPBD;
            window.RTPhysics.projectDistanceXPBD=(...args)=>{const result=xpbd(...args);
              if(corrupt){corrupt=false;result.b.x=Number.NaN;}return result;};
            function canvasDrag(demo,canvas,x,y){
              canvas.pathDraws=[];demo.draw();
              const circles=canvas.pathDraws.filter(call=>call.op==='arc');
              const sphereCircle=circles.reduce((largest,call)=>call.r>largest.r?call:largest);
              canvas.dispatch('pointerdown',{clientX:sphereCircle.x,clientY:sphereCircle.y,pointerId:4});
              canvas.dispatch('pointermove',{clientX:x,clientY:y,pointerId:4});
            }
        """)
        self.assertEqual(data["frames"], 360)
        self.assertTrue(data["finite"])
        self.assertTrue(data["fits"])

    def test_soft_area_uses_real_area_xpbd_and_reports_hard_soft_deformation(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics20.js"], """
            const assert=require('node:assert/strict');
            const demo=mounts[0], root=document.getElementById('soft-area');
            assert.deepEqual(inputs(root).map(input=>Number(input.value)),[0,0.01,8]);
            let state=demo.getState();
            assert.equal(state.mode,'硬'); assert.equal(state.compliance,0);
            assert.equal(state.restArea,3); assert.equal(state.restPerimeter,8);
            assert.equal(state.particles.length,4); assert.equal(state.area,3);
            assert.ok(state.particles.every(p=>p.w===1)); assert.equal(state.lambda,0);
            const initial=JSON.parse(JSON.stringify(state.particles));
            demo.step();
            assert.equal(areaCalls,8,JSON.stringify({state:demo.getState(),
              warning:children(root).find(e=>e.attributes.role==='status').textContent}));
            assert.ok(realCalls&&hardCalls&&lambdaContinuous);
            assert.ok(Math.abs(state.area-3)<1e-8);
            assert.equal(state.absoluteError,Math.abs(state.area-3));
            assert.equal(state.relativeError,state.absoluteError/3);
            assert.ok(state.perimeter>0&&state.perimeterRatio>0);
            const controls=inputs(root);
            controls[0].value='1'; controls[0].dispatch('input');
            state=demo.getState();
            assert.equal(state.mode,'软'); assert.equal(state.frame,0);
            assert.deepEqual(JSON.parse(JSON.stringify(state.particles)),initial);
            demo.step();
            assert.ok(state.area>3); assert.ok(state.relativeError>0);
            assert.ok(state.compliance>0); assert.equal(areaCalls,16);
            const text=readout(root).textContent;
            for(const marker of ['绝对面积误差','相对面积误差','周长','周长比','柔度','迭代','lambda','有限状态'])
              assert.ok(text.includes(marker),marker);
            assert.doesNotMatch(text,/NaN|Infinity/);
            console.log(JSON.stringify({hardArea:3,softArea:state.area,softError:state.relativeError,
              perimeter:state.perimeter,ratio:state.perimeterRatio,areaCalls}));
        """, setup="""
            const area=window.RTPhysics.projectAreaXPBD;
            let areaCalls=0, realCalls=true, hardCalls=true, lambdaContinuous=true;
            let previous=0;
            window.RTPhysics.projectAreaXPBD=(...args)=>{
              areaCalls++;
              const expectedLambda=areaCalls%8===1?0:previous;
              if(args[4]!==3||args[6]!==1/60||args[7]!==expectedLambda) realCalls=false;
              if(areaCalls<=8 && args[5]!==0) hardCalls=false;
              if(areaCalls>8 && !(args[5]>0)) hardCalls=false;
              const result=area(...args); previous=result.lambda;
              lambdaContinuous=lambdaContinuous&&Number.isFinite(result.lambda);
              return result;
            };
        """)
        self.assertEqual(data["areaCalls"], 16)
        self.assertGreater(data["softArea"], 3)
        self.assertGreater(data["ratio"], 1)

    def test_soft_area_drag_rejects_degenerate_crossed_and_nonfinite_geometry_and_finites_reset(self):
        data = self.run_demos([REPO_DIR / "book/assets/demos/physics20.js"], """
            const assert=require('node:assert/strict');
            const demo=mounts[0], root=document.getElementById('soft-area'), canvas=demo.canvas;
            canvas.clientWidth=328;canvas.clientHeight=180;
            demo.step(); assert.notEqual(demo.getState().lambda,0);
            demo.setRunning(true); assert.equal(pending.size,1);
            canvas.pathDraws=[];demo.draw();
            let points=canvas.pathDraws.filter(call=>call.op==='arc');
            canvas.dispatch('pointerdown',{clientX:points[0].x,clientY:points[0].y,pointerId:1});
            assert.equal(pending.size,0);assert.equal(demo.getState().lambda,0);
            canvas.dispatch('pointermove',{clientX:-9999,clientY:-9999,pointerId:1});
            canvas.dispatch('pointerup',{pointerId:1});
            let state=demo.getState();
            assert.ok(state.particles[0].x>=-0.6&&state.particles[0].x<=3.6);
            assert.ok(state.particles[0].y>=-0.6&&state.particles[0].y<=1.6);
            assert.equal(state.particles[0].vx,0);assert.equal(state.particles[0].vy,0);
            assert.equal(state.dragging,null);
            function tryMove(index,target){
              canvas.pathDraws=[];demo.draw();points=canvas.pathDraws.filter(call=>call.op==='arc');
              canvas.dispatch('pointerdown',{clientX:points[index].x,clientY:points[index].y,pointerId:2});
              const before=JSON.parse(JSON.stringify(demo.getState().particles));
              canvas.dispatch('pointermove',{clientX:target.x,clientY:target.y,pointerId:2});
              canvas.dispatch('pointerup',{pointerId:2});
              assert.deepEqual(JSON.parse(JSON.stringify(demo.getState().particles)),before);
              assert.equal(demo.getState().finite,true);assert.equal(demo.getState().invalid,false);
              assert.equal(pending.size,0);
            }
            canvas.pathDraws=[];demo.draw();points=canvas.pathDraws.filter(call=>call.op==='arc');
            tryMove(1,{x:points[0].x,y:points[0].y});
            assert.match(children(root).find(e=>e.attributes.role==='status').textContent,/重合|退化/);
            canvas.pathDraws=[];demo.draw();points=canvas.pathDraws.filter(call=>call.op==='arc');
            const midpoint={x:(points[0].x+points[2].x)/2,y:(points[0].y+points[2].y)/2};
            tryMove(1,midpoint);
            assert.match(children(root).find(e=>e.attributes.role==='status').textContent,/共线|退化/);
            canvas.pathDraws=[];demo.draw();points=canvas.pathDraws.filter(call=>call.op==='arc');
            const rightBottom={x:points[2].x,y:points[0].y+2*(points[2].y-points[0].y)};
            tryMove(0,rightBottom);
            assert.match(children(root).find(e=>e.attributes.role==='status').textContent,/交叉|翻面|退化/);
            demo.reset();demo.step();
            canvas.pathDraws=[];demo.draw();points=canvas.pathDraws.filter(call=>call.op==='arc');
            canvas.dispatch('pointerdown',{clientX:points[0].x,clientY:points[0].y,pointerId:3});
            const before=JSON.parse(JSON.stringify(demo.getState().particles));
            canvas.dispatch('pointermove',{clientX:Number.NaN,clientY:10,pointerId:3});
            canvas.dispatch('pointerup',{pointerId:3});
            assert.deepEqual(JSON.parse(JSON.stringify(demo.getState().particles)),before);
            assert.equal(demo.getState().finite,true);
            demo.reset();
            for(let frame=0;frame<60;frame++) demo.step();
            state=demo.getState();assert.equal(state.frame,60);assert.equal(state.finished,true);
            assert.equal(state.finite,true);assert.ok(Math.abs(state.time-1)<1e-12);assert.equal(pending.size,0);
            demo.reset();demo.step();const first=JSON.parse(JSON.stringify(demo.getState()));
            demo.reset();demo.step();assert.deepEqual(JSON.parse(JSON.stringify(demo.getState())),first);
            demo.reset();corrupt=true;demo.step();state=demo.getState();
            assert.equal(state.invalid,true);assert.equal(state.finite,false);assert.equal(pending.size,0);
            assert.doesNotMatch(readout(root).textContent,/NaN|Infinity/);
            demo.reset();const controls=inputs(root);
            controls[0].value='1';controls[1].value='0';controls[2].value='40';controls[0].dispatch('input');
            assert.equal(demo.getState().invalid,true);assert.equal(demo.getState().mode,'硬');
            demo.reset();assert.deepEqual(inputs(root).map(input=>Number(input.value)),[0,0.01,8]);
            console.log(JSON.stringify({dragPaused:true,clamped:true,degenerateRejected:true,
              crossedRejected:true,nonfiniteRejected:true,frames:60,deterministic:true}));
        """, setup="""
            let corrupt=false;const area=window.RTPhysics.projectAreaXPBD;
            window.RTPhysics.projectAreaXPBD=(...args)=>{const result=area(...args);
              if(corrupt){corrupt=false;result.particles[0].x=Number.NaN;}return result;};
        """)
        self.assertEqual(data["frames"], 60)
        self.assertTrue(data["degenerateRejected"])
        self.assertTrue(data["crossedRejected"])
        self.assertTrue(data["nonfiniteRejected"])

    def test_chapter_ten_demos_are_independent_reduced_motion_safe_and_fit_narrow_canvas(self):
        data = self.run_demos([
            REPO_DIR / "book/assets/demos/physics19.js", REPO_DIR / "book/assets/demos/physics20.js",
        ], """
            const assert=require('node:assert/strict');
            const cloth=mounts[0], area=mounts[1];
            assert.equal(mounts.length,2);assert.notEqual(cloth.canvas,area.canvas);
            assert.equal(pending.size,0);cloth.step();assert.equal(area.getState().frame,0);
            area.step();assert.equal(cloth.getState().frame,1);
            for(const demo of [cloth,area]){
              demo.canvas.clientWidth=328;demo.canvas.clientHeight=180;
              demo.canvas.pathDraws=[];demo.canvas.textDraws=[];demo.draw();
              assert.equal(demo.canvas.width,328);assert.equal(demo.canvas.height,180);
              for(const call of demo.canvas.pathDraws){
                assert.ok(Number.isFinite(call.x)&&Number.isFinite(call.y));
                assert.ok(call.x>=0&&call.x<=328&&call.y>=0&&call.y<=180);
                if(call.op==='arc')assert.ok(call.x-call.r>=0&&call.x+call.r<=328&&
                  call.y-call.r>=0&&call.y+call.r<=180);
              }
              for(const call of demo.canvas.textDraws)
                assert.ok(call.x>=0&&call.x+call.text.length*13<=328&&call.y>=0&&call.y<=180,call.text);
            }
            cloth.setRunning(true);area.setRunning(true);assert.equal(pending.size,2);
            cloth.reset();assert.equal(pending.size,1);area.reset();assert.equal(pending.size,0);
            assert.match(readout(document.getElementById('cloth-showcase')).textContent,/固定点数/);
            assert.match(readout(document.getElementById('soft-area')).textContent,/二维.*面积类比/);
            console.log(JSON.stringify({independent:true,reducedMotion:true,width:328,height:180}));
        """, setup="""
            window.matchMedia=()=>({matches:true});
        """)
        self.assertEqual(data, {"independent": True, "reducedMotion": True,
                                "width": 328, "height": 180})


class PhysicsCoreChapterEightTests(unittest.TestCase):
    def run_node(self, script):
        result = subprocess.run(
            ["node", "-e", textwrap.dedent(script), str(CORE_PATH)],
            cwd=REPO_DIR, capture_output=True, text=True, timeout=10, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        self.assertEqual(result.stderr, "")
        return json.loads(result.stdout)

    def test_nonzero_compliance_xpbd_is_substep_invariant_within_stated_tolerance(self):
        data = self.run_node("""
            const assert = require('node:assert/strict');
            const P = require(process.argv[1]);
            const compliance = 0.01;
            function solve(substeps) {
                let pair = [{x: 0, y: 0, w: 1}, {x: 3, y: 0, w: 1}];
                const h = (1 / 60) / substeps;
                for (let step = 0; step < substeps; step += 1) {
                    let lambda = 0;
                    for (let iteration = 0; iteration < 40; iteration += 1) {
                        const result = P.projectDistanceXPBD(
                            pair[0], pair[1], 2, compliance, h, lambda);
                        pair = [result.a, result.b];
                        lambda = result.lambda;
                    }
                }
                return {length: Math.hypot(pair[1].x - pair[0].x, pair[1].y - pair[0].y),
                    pair, h};
            }
            const one = solve(1), two = solve(2);
            const difference = Math.abs(one.length - two.length);
            assert.ok(compliance > 0, 'compliance must be nonzero');
            assert.equal(one.h, 1 / 60);
            assert.equal(two.h, 1 / 120);
            assert.ok(difference < 0.03, 'stated final-length tolerance is 0.03 m');
            assert.ok(one.length > 2 && two.length > 2, 'positive compliance leaves residual error');
            console.log(JSON.stringify({one, two, difference, compliance, tolerance: 0.03}));
        """)
        self.assertEqual(data["compliance"], 0.01)
        self.assertEqual(data["tolerance"], 0.03)
        self.assertLess(data["difference"], 0.03)
        self.assertGreater(data["difference"], 0.02)
        self.assertGreater(data["one"]["length"], 2)
        self.assertGreater(data["two"]["length"], 2)


class PhysicsUITests(unittest.TestCase):
    def run_ui(self, script):
        result = subprocess.run(
            ["node", "-e", textwrap.dedent(script), str(UI_PATH)],
            cwd=REPO_DIR, capture_output=True, text=True, timeout=10, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        self.assertEqual(result.stderr, "")
        return json.loads(result.stdout)

    def run_raf_ui(self, script):
        return self.run_ui("""
            const assert = require('node:assert/strict');
            const fs = require('node:fs');
            const vm = require('node:vm');
            class Element {
              constructor(tag) {
                this.tagName = tag; this.children = []; this.attributes = {}; this.listeners = {};
              }
              appendChild(child) { this.children.push(child); child.parent = this; return child; }
              append(...children) { children.forEach(child => this.appendChild(child)); }
              replaceChildren() { this.children = []; }
              setAttribute(key, value) { this.attributes[key] = value; }
              addEventListener(key, fn) { (this.listeners[key] ||= []).push(fn); }
              getContext() { return new Proxy({}, {get: () => () => {}}); }
              getBoundingClientRect() { return {left: 0, top: 0, width: 200, height: 100}; }
            }
            const root = new Element('div'); root.id = 'demo';
            const pending = new Map();
            let nextFrameId = 1;
            const requested = [];
            const cancelled = [];
            const window = {
              devicePixelRatio: 1, matchMedia: () => ({matches: false}),
              requestAnimationFrame: callback => {
                const id = nextFrameId++; pending.set(id, callback); requested.push(id); return id;
              },
              cancelAnimationFrame: id => { cancelled.push(id); pending.delete(id); }
            };
            window.Element = Element;
            window.addEventListener = () => {};
            window.document = {
              getElementById: id => id === 'demo' ? root : null,
              createElement: tag => new Element(tag)
            };
            function tick(time) {
              const [id, callback] = pending.entries().next().value || [];
              if (id === undefined) return false;
              pending.delete(id);
              callback(time);
              return true;
            }
            vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), {window, document: window.document});
        """ + textwrap.dedent(script))

    def test_mount_controls_reset_and_finite_readouts(self):
        data = self.run_ui("""
            const assert = require('node:assert/strict');
            const fs = require('node:fs');
            const vm = require('node:vm');
            class Element {
              constructor(tag) {
                this.tagName = tag; this.children = []; this.attributes = {}; this.listeners = {};
                this.value = ''; this.widthAssignments = 0; this.heightAssignments = 0;
                if (tag === 'canvas') {
                  let width = 0; let height = 0;
                  Object.defineProperty(this, 'width', {
                    get: () => width, set: value => { width = value; this.widthAssignments++; }
                  });
                  Object.defineProperty(this, 'height', {
                    get: () => height, set: value => { height = value; this.heightAssignments++; }
                  });
                }
              }
              appendChild(child) { this.children.push(child); child.parent = this; return child; }
              append(...children) { children.forEach(child => this.appendChild(child)); }
              replaceChildren() { this.children = []; }
              setAttribute(key, value) { this.attributes[key] = value; }
              addEventListener(key, fn) { (this.listeners[key] ||= []).push(fn); }
              dispatch(key) { for (const fn of this.listeners[key] || []) fn({target: this}); }
              getContext() {
                return new Proxy({}, {get: (_, key) => key === 'setTransform'
                  ? (...args) => { this.transforms.push(args); } : () => {}});
              }
              getBoundingClientRect() { return {left: 10, top: 20, width: 320, height: 180}; }
            }
            const roots = {first: new Element('div')};
            const context = new Proxy({}, {get: () => () => {}});
            const window = {devicePixelRatio: 2, matchMedia: () => ({matches: false}),
              requestAnimationFrame: () => 1, cancelAnimationFrame: () => {}};
            window.Element = Element;
            window.addEventListener = () => {};
            window.document = {getElementById: id => roots[id], createElement: tag => {
              const element = new Element(tag); element.transforms = []; return element;
            }};
            window.HTMLCanvasElement = Element;
            vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), {
              window, document: {getElementById: id => roots[id], createElement: tag => new Element(tag)}
            });
            const first = window.RTPhysicsUI.mount('first', {
              createState: () => ({x: 0}),
              controls: [{key: 'v', label: '速度', min: 0, max: 10, step: 1, value: 2}],
              update: (state, dt, values) => ({x: state.x + values.v * dt}),
              draw: (ctx, state, width, height) => { ctx.dot(state.x, width, height); },
              metrics: state => ({位置: state.x})
            });
            const descendants = element => element.children.flatMap(child => [child, ...descendants(child)]);
            const slider = descendants(first.canvas.parent).find(el => el.tagName === 'input');
            slider.value = '4'; slider.dispatch('input');
            assert.equal(first.getState().x, 0);
            assert.equal(descendants(first.canvas.parent).find(el => el.tagName === 'output').textContent, '4');
            first.step();
            assert.ok(Math.abs(first.getState().x - 4 / 60) < 1e-12);
            first.reset();
            assert.equal(first.getState().x, 0);
            assert.equal(Number(slider.value), 2);
            assert.equal(first.canvas.width, 640);
            assert.equal(first.canvas.height, 360);
            const widthAssignments = first.canvas.widthAssignments;
            const heightAssignments = first.canvas.heightAssignments;
            first.draw();
            first.draw();
            assert.equal(first.canvas.widthAssignments, widthAssignments);
            assert.equal(first.canvas.heightAssignments, heightAssignments);
            window.devicePixelRatio = 3;
            first.draw();
            assert.equal(first.canvas.width, 960);
            assert.equal(first.canvas.height, 540);
            assert.equal(first.canvas.widthAssignments, widthAssignments + 1);
            assert.equal(first.canvas.heightAssignments, heightAssignments + 1);
            assert.deepEqual(first.canvas.transforms.at(-1), [3, 0, 0, 3, 0, 0]);
            assert.ok(first.canvas.parent.children.some(el => el.attributes['data-readout'] !== undefined));
            assert.equal(window.RTPhysicsUI.canvasPoint(first.canvas, {clientX: 170, clientY: 110}).x, 160);
            console.log(JSON.stringify({ok: true, state: first.getState(), value: slider.value}));
        """)
        self.assertEqual(data, {"ok": True, "state": {"x": 0}, "value": 2})

    def test_controllers_are_independent_and_reduced_motion_allows_step(self):
        data = self.run_ui("""
            const assert = require('node:assert/strict');
            const fs = require('node:fs');
            const vm = require('node:vm');
            class Element {
              constructor(tag) { this.tagName = tag; this.children = []; this.attributes = {}; this.listeners = {}; this.classList = {add() {}}; }
              appendChild(child) { this.children.push(child); child.parent = this; return child; }
              append(...children) { children.forEach(child => this.appendChild(child)); }
              replaceChildren() { this.children = []; }
              setAttribute(key, value) { this.attributes[key] = value; }
              addEventListener(key, fn) { (this.listeners[key] ||= []).push(fn); }
              getContext() { return new Proxy({}, {get: () => () => {}}); }
              getBoundingClientRect() { return {left: 0, top: 0, width: 200, height: 100}; }
            }
            const roots = {a: new Element('div'), b: new Element('div')};
            const window = {devicePixelRatio: 1, matchMedia: () => ({matches: true}),
              requestAnimationFrame: () => { throw new Error('reduced motion must stay paused'); },
              cancelAnimationFrame: () => {}};
            window.Element = Element;
            window.addEventListener = () => {};
            window.document = {getElementById: id => roots[id], createElement: tag => new Element(tag)};
            vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), {
              window, document: {getElementById: id => roots[id], createElement: tag => new Element(tag)}
            });
            const options = {
              createState: () => ({time: 0}), controls: [],
              update: (state, dt) => ({time: state.time + dt}), draw: () => {},
              metrics: state => ({时间: state.time})
            };
            const a = window.RTPhysicsUI.mount('a', options);
            const b = window.RTPhysicsUI.mount('b', options);
            const toolbar = a.canvas.parent.children.find(el => el.className === 'demo-toolbar physics-controls');
            assert.equal(toolbar.children[1].textContent, '运行');
            b.step();
            window.RTPhysicsUI.step();
            assert.equal(a.getState().time, 0);
            assert.ok(Math.abs(b.getState().time - 2 / 60) < 1e-12);
            assert.notEqual(a.canvas, b.canvas);
            console.log(JSON.stringify({a: a.getState(), b: b.getState()}));
        """)
        self.assertEqual(data["a"]["time"], 0)
        self.assertAlmostEqual(data["b"]["time"], 2 / 60)

    def test_non_finite_automatic_update_pauses_and_requests_reset(self):
        data = self.run_raf_ui("""
            const updates = [];
            const controller = window.RTPhysicsUI.mount('demo', {
              createState: () => ({x: 0}), controls: [],
              update: (state, dt) => { updates.push(dt); return {x: Infinity}; },
              draw: () => {}, metrics: () => ({位置: 1})
            });
            const toolbar = root.children.find(el => el.className === 'demo-toolbar physics-controls');
            const readout = root.children.find(el => el.attributes['data-readout'] !== undefined);
            assert.equal(pending.size, 1);
            tick(0);
            assert.equal(updates.length, 0);
            tick(1000 / 60);
            assert.deepEqual(updates, [1 / 60]);
            assert.equal(pending.size, 0);
            assert.equal(toolbar.children[1].textContent, '运行');
            assert.match(readout.textContent, /重置/);
            assert.doesNotMatch(readout.textContent, /Infinity|NaN/);
            const message = readout.textContent;
            const requestsAfterFailure = requested.length;
            tick(2000);
            controller.setRunning(true);
            assert.equal(requested.length, requestsAfterFailure);
            assert.equal(pending.size, 0);
            controller.reset();
            assert.equal(controller.getState().x, 0);
            assert.equal(readout.textContent, '位置：1');
            console.log(JSON.stringify({updates, message, requests: requested.length}));
        """)
        self.assertEqual(data["updates"], [1 / 60])
        self.assertIn("重置", data["message"])
        self.assertEqual(data["requests"], 2)

    def test_raf_uses_fixed_sixtieth_second_updates(self):
        data = self.run_raf_ui("""
            const updates = [];
            const controller = window.RTPhysicsUI.mount('demo', {
              createState: () => ({time: 0}), controls: [],
              update: (state, dt) => { updates.push(dt); return {time: state.time + dt}; },
              draw: () => {}, metrics: state => ({时间: state.time})
            });
            tick(0);
            tick(1000 / 60);
            tick(2000 / 60);
            assert.deepEqual(updates, [1 / 60, 1 / 60]);
            assert.ok(Math.abs(controller.getState().time - 2 / 60) < 1e-12);
            const frameId = pending.keys().next().value;
            controller.setRunning(false);
            assert.deepEqual(cancelled, [frameId]);
            assert.equal(pending.size, 0);
            console.log(JSON.stringify({updates, time: controller.getState().time}));
        """)
        self.assertEqual(data["updates"], [1 / 60, 1 / 60])
        self.assertAlmostEqual(data["time"], 2 / 60)

    def test_raf_bounds_catch_up_to_quarter_second(self):
        data = self.run_raf_ui("""
            const updates = [];
            const controller = window.RTPhysicsUI.mount('demo', {
              createState: () => ({time: 0}), controls: [],
              update: (state, dt) => { updates.push(dt); return {time: state.time + dt}; },
              draw: () => {}, metrics: state => ({时间: state.time})
            });
            tick(0);
            tick(5000);
            assert.equal(updates.length, 15);
            assert.ok(updates.every(dt => dt === 1 / 60));
            assert.ok(Math.abs(controller.getState().time - 0.25) < 1e-12);
            tick(5250);
            assert.equal(updates.length, 30);
            assert.ok(Math.abs(controller.getState().time - 0.5) < 1e-12);
            controller.setRunning(false);
            console.log(JSON.stringify({updates: updates.length, time: controller.getState().time}));
        """)
        self.assertEqual(data["updates"], 30)
        self.assertAlmostEqual(data["time"], 0.5)


class PhysicsCoreTests(unittest.TestCase):
    def run_node(self, script):
        result = subprocess.run(
            ["node", "-e", textwrap.dedent(script), str(CORE_PATH)],
            cwd=REPO_DIR,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
        self.assertEqual(result.stderr, "")
        return json.loads(result.stdout)

    def test_vector_operations_and_zero_normalization(self):
        data = self.run_node("""
            const P = require(process.argv[1]);
            const a = Object.freeze({x: 3, y: 4});
            const b = Object.freeze({x: -1, y: 2});
            const sum = P.vec2.add(a, b);
            const difference = P.vec2.sub(a, b);
            const scaled = P.vec2.scale(a, -2);
            const unit = P.vec2.normalize(a);
            const zero = P.vec2.normalize({x: 0, y: 0});
            console.log(JSON.stringify({
                sum, difference, scaled,
                dot: P.vec2.dot(a, b), length: P.vec2.length(a),
                unit, zero, original: a
            }));
        """)
        self.assertEqual(data["sum"], {"x": 2, "y": 6})
        self.assertEqual(data["difference"], {"x": 4, "y": 2})
        self.assertEqual(data["scaled"], {"x": -6, "y": -8})
        self.assertEqual(data["dot"], 5)
        self.assertEqual(data["length"], 5)
        self.assertEqual(data["unit"], {"x": 0.6, "y": 0.8})
        self.assertEqual(data["zero"], {"x": 0, "y": 0})
        self.assertEqual(data["original"], {"x": 3, "y": 4})

    def test_projectile_integration_methods(self):
        data = self.run_node("""
            const assert = require('node:assert/strict');
            const P = require(process.argv[1]);
            const exactInput = Object.freeze({x: 0, y: 0, vx: 10, vy: 20});
            const exact = P.integrateProjectile(
                exactInput, {x: 0, y: -9.81}, 1, 'exact'
            );
            assert.notStrictEqual(exact, exactInput);
            assert.ok(Math.abs(exact.x - 10) < 1e-12);
            assert.ok(Math.abs(exact.y - 15.095) < 1e-12);
            const input = Object.freeze({x: 1, y: 2, vx: 3, vy: 4, w: 0});
            const gravity = Object.freeze({x: 0, y: -10});
            const forward = P.integrateProjectile(input, gravity, 0.5, 'forward');
            const semiImplicit = P.integrateProjectile(input, gravity, 0.5, 'semi-implicit');
            const unchanged = P.integrateProjectile(input, gravity, 0, 'exact');
            assert.notStrictEqual(forward, input);
            assert.notStrictEqual(semiImplicit, input);
            assert.notStrictEqual(unchanged, input);
            console.log(JSON.stringify({exact, forward, semiImplicit, unchanged, exactInput, input}));
        """)
        self.assertAlmostEqual(data["exact"]["x"], 10, places=14)
        self.assertAlmostEqual(data["exact"]["y"], 15.095, places=14)
        self.assertEqual(data["exact"]["vx"], 10)
        self.assertEqual(data["exact"]["vy"], 10.19)
        self.assertEqual(data["exactInput"], {"x": 0, "y": 0, "vx": 10, "vy": 20})
        self.assertEqual(data["forward"], {"x": 2.5, "y": 4, "vx": 3, "vy": -1, "w": 0})
        self.assertEqual(data["semiImplicit"], {"x": 2.5, "y": 1.5, "vx": 3, "vy": -1, "w": 0})
        self.assertEqual(data["unchanged"], {"x": 1, "y": 2, "vx": 3, "vy": 4, "w": 0})
        self.assertEqual(data["input"], {"x": 1, "y": 2, "vx": 3, "vy": 4, "w": 0})

    def test_exact_projectile_repeated_equal_steps_match_one_step(self):
        data = self.run_node("""
            const assert = require('node:assert/strict');
            const P = require(process.argv[1]);
            const input = Object.freeze({x: 1, y: 2, vx: 3, vy: 4, w: 0});
            const gravity = Object.freeze({x: 2, y: -10});
            const oneStep = P.integrateProjectile(input, gravity, 1, 'exact');
            const firstHalf = P.integrateProjectile(input, gravity, 0.5, 'exact');
            const twoSteps = P.integrateProjectile(firstHalf, gravity, 0.5, 'exact');
            for (const key of ['x', 'y', 'vx', 'vy']) {
              assert.ok(Math.abs(twoSteps[key] - oneStep[key]) < 1e-12, key);
            }
            assert.notStrictEqual(twoSteps, firstHalf);
            console.log(JSON.stringify({oneStep, twoSteps, input, gravity}));
        """)
        expected = {"x": 5, "y": 1, "vx": 5, "vy": -6, "w": 0}
        for key, value in expected.items():
            self.assertAlmostEqual(data["oneStep"][key], value, places=14)
            self.assertAlmostEqual(data["twoSteps"][key], value, places=14)
        self.assertEqual(data["input"], {"x": 1, "y": 2, "vx": 3, "vy": 4, "w": 0})
        self.assertEqual(data["gravity"], {"x": 2, "y": -10})

    def test_exact_projectile_repeated_unequal_steps_accumulate_both_components(self):
        data = self.run_node("""
            const assert = require('node:assert/strict');
            const P = require(process.argv[1]);
            const input = Object.freeze({x: 0, y: 0, vx: 10, vy: 20});
            const gravity = Object.freeze({x: 2, y: -9.81});
            const first = P.integrateProjectile(input, gravity, 0.25, 'exact');
            const repeated = P.integrateProjectile(first, gravity, 0.75, 'exact');
            const oneStep = P.integrateProjectile(input, gravity, 1, 'exact');
            for (const key of ['x', 'y', 'vx', 'vy']) {
              assert.ok(Math.abs(repeated[key] - oneStep[key]) < 1e-12, key);
            }
            console.log(JSON.stringify({repeated, oneStep, input, gravity}));
        """)
        expected = {"x": 11, "y": 15.095, "vx": 12, "vy": 10.19}
        for key, value in expected.items():
            self.assertAlmostEqual(data["repeated"][key], value, places=14)
            self.assertAlmostEqual(data["oneStep"][key], value, places=14)
        self.assertEqual(data["input"], {"x": 0, "y": 0, "vx": 10, "vy": 20})
        self.assertEqual(data["gravity"], {"x": 2, "y": -9.81})

    def test_exact_projectile_horizontal_acceleration_with_zero_vertical_gravity(self):
        data = self.run_node("""
            const assert = require('node:assert/strict');
            const P = require(process.argv[1]);
            const input = Object.freeze({x: 1, y: 2, vx: 3, vy: 4, w: 0});
            const gravity = Object.freeze({x: 4, y: 0});
            const exact = P.integrateProjectile(input, gravity, 0.5, 'exact');
            assert.notStrictEqual(exact, input);
            console.log(JSON.stringify({exact, input, gravity}));
        """)
        self.assertEqual(data["exact"], {"x": 3, "y": 4, "vx": 5, "vy": 4, "w": 0})
        self.assertEqual(data["input"], {"x": 1, "y": 2, "vx": 3, "vy": 4, "w": 0})
        self.assertEqual(data["gravity"], {"x": 4, "y": 0})

    def test_spring_force_energy_and_stable_step(self):
        data = self.run_node("""
            const P = require(process.argv[1]);
            console.log(JSON.stringify({
                stretched: P.springForce(2, 3, 10, 4),
                compressed: P.springForce(-2, 0, 10, 4),
                equilibrium: P.springForce(0, 0, 10, 4),
                energy: P.energy1d(2, 3, 5, 4),
                stable: P.stableSpringStep(100, 4)
            }));
        """)
        self.assertEqual(data["stretched"], -32)
        self.assertEqual(data["compressed"], 20)
        self.assertEqual(data["equilibrium"], 0)
        self.assertEqual(data["energy"], {"kinetic": 9, "potential": 40})
        self.assertEqual(data["stable"], 0.4)

    def test_pbd_single_projection_and_purity(self):
        data = self.run_node("""
            const assert = require('node:assert/strict');
            const P = require(process.argv[1]);
            const a = Object.freeze({x: 0, y: 0, w: 1});
            const b = Object.freeze({x: 3, y: 0, w: 1});
            const pbd = P.projectDistance(a, b, 2, 1);
            assert.ok(Math.abs(pbd.a.x - 0.5) < 1e-12);
            assert.ok(Math.abs(pbd.b.x - 2.5) < 1e-12);
            assert.ok(Math.abs(P.vec2.length(P.vec2.sub(pbd.b, pbd.a)) - 2) < 1e-12);
            const fixed = P.projectDistance({x: 0, y: 0, w: 0}, b, 2, 1);
            assert.equal(fixed.a.x, 0);
            assert.ok(Math.abs(fixed.b.x - 2) < 1e-12);
            const half = P.projectDistance(a, b, 2, 0.5);
            const compressed = P.projectDistance(a, {x: 1, y: 0, w: 1}, 2, 1);
            const bothFixed = P.projectDistance(
                {x: 0, y: 0, w: 0}, {x: 3, y: 0, w: 0}, 2, 1
            );
            const coincidentInput = Object.freeze({x: 4, y: 5, w: 1});
            const coincident = P.projectDistance(coincidentInput, coincidentInput, 2, 1);
            assert.notStrictEqual(pbd.a, a);
            assert.notStrictEqual(pbd.b, b);
            assert.notStrictEqual(coincident.a, coincidentInput);
            assert.notStrictEqual(coincident.b, coincidentInput);
            console.log(JSON.stringify({pbd, fixed, half, compressed, bothFixed, coincident, a, b}));
        """)
        self.assertEqual(data["pbd"], {
            "a": {"x": 0.5, "y": 0, "w": 1},
            "b": {"x": 2.5, "y": 0, "w": 1},
        })
        self.assertEqual(data["fixed"], {
            "a": {"x": 0, "y": 0, "w": 0},
            "b": {"x": 2, "y": 0, "w": 1},
        })
        self.assertEqual(data["half"], {
            "a": {"x": 0.25, "y": 0, "w": 1},
            "b": {"x": 2.75, "y": 0, "w": 1},
        })
        self.assertEqual(data["compressed"], {
            "a": {"x": -0.5, "y": 0, "w": 1},
            "b": {"x": 1.5, "y": 0, "w": 1},
        })
        self.assertEqual(data["bothFixed"], {
            "a": {"x": 0, "y": 0, "w": 0},
            "b": {"x": 3, "y": 0, "w": 0},
        })
        self.assertEqual(data["coincident"], {
            "a": {"x": 4, "y": 5, "w": 1},
            "b": {"x": 4, "y": 5, "w": 1},
        })
        self.assertEqual(data["a"], {"x": 0, "y": 0, "w": 1})
        self.assertEqual(data["b"], {"x": 3, "y": 0, "w": 1})

    def test_xpbd_hand_derived_compliance_and_lambda_history(self):
        data = self.run_node("""
            const assert = require('node:assert/strict');
            const P = require(process.argv[1]);
            // alpha_tilde = 1 / 0.5^2 = 4; delta_lambda = -1 / (2 + 4) = -1/6.
            // a moves by -delta_lambda = +1/6; b moves by delta_lambda = -1/6.
            const a = Object.freeze({x: 0, y: 0, w: 1});
            const b = Object.freeze({x: 3, y: 0, w: 1});
            const soft = P.projectDistanceXPBD(a, b, 2, 1, 0.5, 0);
            assert.notStrictEqual(soft.a, a);
            assert.notStrictEqual(soft.b, b);
            // delta_lambda = (-1 - 4 * 0.25) / 6 = -1/3; lambda' = -1/12.
            const history = P.projectDistanceXPBD(a, b, 2, 1, 0.5, 0.25);
            console.log(JSON.stringify({soft, history}));
        """)
        soft_a, soft_b = data["soft"]["a"], data["soft"]["b"]
        self.assertAlmostEqual(soft_a["x"], 1 / 6, places=14)
        self.assertAlmostEqual(soft_a["y"], 0, places=14)
        self.assertEqual(soft_a["w"], 1)
        self.assertAlmostEqual(soft_b["x"], 3 - 1 / 6, places=14)
        self.assertAlmostEqual(soft_b["y"], 0, places=14)
        self.assertEqual(soft_b["w"], 1)
        self.assertAlmostEqual(data["soft"]["lambda"], -1 / 6, places=14)
        history_a, history_b = data["history"]["a"], data["history"]["b"]
        self.assertAlmostEqual(history_a["x"], 1 / 3, places=14)
        self.assertAlmostEqual(history_b["x"], 3 - 1 / 3, places=14)
        self.assertAlmostEqual(data["history"]["lambda"], -1 / 12, places=14)

    def test_xpbd_zero_compliance_matches_pbd(self):
        data = self.run_node("""
            const assert = require('node:assert/strict');
            const P = require(process.argv[1]);
            const a = {x: 0, y: 0, w: 1};
            const b = {x: 3, y: 0, w: 1};
            const pbd = P.projectDistance(a, b, 2, 1);
            const xpbd = P.projectDistanceXPBD(a, b, 2, 0, 1 / 60, 0);
            assert.deepEqual(xpbd.a, pbd.a);
            assert.deepEqual(xpbd.b, pbd.b);
            assert.equal(xpbd.lambda, -0.5);
            console.log(JSON.stringify({xpbd}));
        """)
        self.assertEqual(data["xpbd"], {
            "a": {"x": 0.5, "y": 0, "w": 1},
            "b": {"x": 2.5, "y": 0, "w": 1},
            "lambda": -0.5,
        })

    def test_xpbd_compliance_direction_and_boundaries(self):
        data = self.run_node("""
            const assert = require('node:assert/strict');
            const P = require(process.argv[1]);
            const a = {x: 0, y: 0, w: 1};
            const b = {x: 3, y: 0, w: 1};
            const soft = P.projectDistanceXPBD(a, b, 2, 1, 1 / 60, 0);
            const hard = P.projectDistanceXPBD(a, b, 2, 0, 1 / 60, 0);
            assert.ok(Math.abs((soft.b.x - soft.a.x) - 2) > Math.abs((hard.b.x - hard.a.x) - 2));
            const fixed = P.projectDistanceXPBD(
                {x: 0, y: 0, w: 0}, b, 2, 0, 1 / 60, 0
            );
            const bothFixed = P.projectDistanceXPBD(
                {x: 0, y: 0, w: 0}, {x: 3, y: 0, w: 0}, 2, 1, 1 / 60, 0
            );
            const coincident = P.projectDistanceXPBD(a, a, 2, 1, 1 / 60, 0);
            console.log(JSON.stringify({
                softError: Math.abs((soft.b.x - soft.a.x) - 2),
                hardError: Math.abs((hard.b.x - hard.a.x) - 2),
                fixed, bothFixed, coincident
            }));
        """)
        self.assertGreater(data["softError"], data["hardError"])
        self.assertEqual(data["fixed"], {
            "a": {"x": 0, "y": 0, "w": 0},
            "b": {"x": 2, "y": 0, "w": 1},
            "lambda": -1,
        })
        self.assertEqual(data["bothFixed"], {
            "a": {"x": 0, "y": 0, "w": 0},
            "b": {"x": 3, "y": 0, "w": 0},
            "lambda": 0,
        })
        self.assertEqual(data["coincident"], {
            "a": {"x": 0, "y": 0, "w": 1},
            "b": {"x": 0, "y": 0, "w": 1},
            "lambda": 0,
        })
        values = [data["softError"], data["hardError"], *self.flatten(data)]
        self.assertTrue(all(isinstance(value, (int, float)) and abs(value) < float("inf")
                            for value in values))

    def test_xpbd_zero_dt_returns_finite_clones_and_preserves_lambda(self):
        data = self.run_node("""
            const assert = require('node:assert/strict');
            const P = require(process.argv[1]);
            const a = Object.freeze({x: 0, y: 0, w: 1});
            const b = Object.freeze({x: 3, y: 0, w: 1});
            const zeroDistance = P.projectDistanceXPBD(a, b, 2, 0, 0, 0.25);
            assert.deepEqual(zeroDistance.a, a);
            assert.deepEqual(zeroDistance.b, b);
            assert.notStrictEqual(zeroDistance.a, a);
            assert.notStrictEqual(zeroDistance.b, b);
            assert.equal(zeroDistance.lambda, 0.25);
            assert.ok(Number.isFinite(zeroDistance.a.x));
            assert.ok(Number.isFinite(zeroDistance.a.y));
            assert.ok(Number.isFinite(zeroDistance.b.x));
            assert.ok(Number.isFinite(zeroDistance.b.y));
            console.log(JSON.stringify({zeroDistance, a, b}));
        """)
        self.assertEqual(data["zeroDistance"], {
            "a": {"x": 0, "y": 0, "w": 1},
            "b": {"x": 3, "y": 0, "w": 1},
            "lambda": 0.25,
        })
        self.assertEqual(data["a"], {"x": 0, "y": 0, "w": 1})
        self.assertEqual(data["b"], {"x": 3, "y": 0, "w": 1})

    def test_xpbd_chapter_worked_example_and_area_quadrilateral_numbers(self):
        data = self.run_node("""
            const assert = require('node:assert/strict');
            const P = require(process.argv[1]);
            const a = Object.freeze({x: 0, y: 0, w: 1});
            const b = Object.freeze({x: 3, y: 0, w: 1});
            // alpha_tilde = 0.01 / (1/60)^2 = 36 and delta_lambda = -1/(2+36).
            const distance = P.projectDistanceXPBD(a, b, 2, 0.01, 1 / 60, 0);
            assert.ok(Math.abs(distance.a.x - 1 / 38) < 1e-14);
            assert.ok(Math.abs(distance.b.x - (3 - 1 / 38)) < 1e-14);
            assert.ok(Math.abs(distance.lambda + 1 / 38) < 1e-14);
            assert.ok(Math.abs((distance.b.x - distance.a.x) - 56 / 19) < 1e-14);

            // Rectangle area is 3, rest area is 2; weighted gradient norm squared is 10.
            // Therefore delta_lambda = -1/(10+36) = -1/46.
            const corners = [
                Object.freeze({x: 0, y: 0, w: 1}), Object.freeze({x: 3, y: 0, w: 1}),
                Object.freeze({x: 3, y: 1, w: 1}), Object.freeze({x: 0, y: 1, w: 1}),
            ];
            const area = P.projectAreaXPBD(...corners, 2, 0.01, 1 / 60, 0);
            assert.ok(Math.abs(area.lambda + 1 / 46) < 1e-14);
            assert.ok(Math.abs(area.particles[0].x - 1 / 92) < 1e-14);
            assert.ok(Math.abs(area.particles[0].y - 3 / 92) < 1e-14);
            assert.ok(Math.abs(area.particles[1].x - (3 - 1 / 92)) < 1e-14);
            assert.ok(Math.abs(area.particles[1].y - 3 / 92) < 1e-14);
            const reversed = [...corners].reverse();
            const clockwise = P.projectAreaXPBD(...reversed, 2, 0.01, 1 / 60, 0);
            assert.ok(Math.abs(clockwise.lambda - 1 / 46) < 1e-14,
                'original winding must normalize the signed area constraint');
            const zeroDt = P.projectAreaXPBD(...corners, 2, 0.01, 0, 0.25);
            assert.equal(zeroDt.lambda, 0.25);
            assert.notStrictEqual(zeroDt.particles[0], corners[0]);
            console.log(JSON.stringify({distance, area: area.particles, areaLambda: area.lambda,
                clockwiseLambda: clockwise.lambda, zeroDtLambda: zeroDt.lambda}));
        """)
        self.assertAlmostEqual(data["distance"]["lambda"], -1 / 38, places=14)
        self.assertAlmostEqual(data["areaLambda"], -1 / 46, places=14)
        self.assertAlmostEqual(data["clockwiseLambda"], 1 / 46, places=14)
        self.assertEqual(data["zeroDtLambda"], 0.25)

    def test_sphere_plane_impact_restitution_and_friction(self):
        data = self.run_node("""
            const assert = require('node:assert/strict');
            const P = require(process.argv[1]);
            const input = Object.freeze({x: 0, y: 0.2, vx: 4, vy: -5, w: 1});
            const hit = P.collideSpherePlane(input, 0.5, 0, 0.5, 0.2);
            assert.equal(hit.y, 0.5);
            assert.equal(hit.vy, 2.5);
            assert.equal(hit.vx, 3.2);
            assert.notStrictEqual(hit, input);
            console.log(JSON.stringify({hit, input}));
        """)
        self.assertEqual(data["hit"], {"x": 0, "y": 0.5, "vx": 3.2, "vy": 2.5, "w": 1})
        self.assertEqual(data["input"], {"x": 0, "y": 0.2, "vx": 4, "vy": -5, "w": 1})

    def test_sphere_plane_separating_above_and_fixed_boundaries(self):
        data = self.run_node("""
            const P = require(process.argv[1]);
            const separating = P.collideSpherePlane(
                {x: 0, y: 0.5, vx: 4, vy: 3, w: 1}, 0.5, 0, 0.5, 0.2
            );
            const deepSeparating = P.collideSpherePlane(
                {x: 0, y: 0.1, vx: 4, vy: 3, w: 1}, 0.5, 0, 0.5, 0.2
            );
            const above = P.collideSpherePlane(
                {x: 2, y: 2, vx: 4, vy: -5, w: 1}, 0.5, 0, 0.5, 0.2
            );
            const resting = P.collideSpherePlane(
                {x: 0, y: 0.5, vx: 4, vy: 0, w: 1}, 0.5, 0, 0.5, 0.2
            );
            const fixed = P.collideSpherePlane(
                {x: 0, y: 0.2, vx: 4, vy: -5, w: 0}, 0.5, 0, 0.5, 0.2
            );
            console.log(JSON.stringify({separating, deepSeparating, above, resting, fixed}));
        """)
        self.assertEqual(data["separating"], {"x": 0, "y": 0.5, "vx": 4, "vy": 3, "w": 1})
        self.assertEqual(data["deepSeparating"], {"x": 0, "y": 0.5, "vx": 4, "vy": 3, "w": 1})
        self.assertEqual(data["above"], {"x": 2, "y": 2, "vx": 4, "vy": -5, "w": 1})
        self.assertEqual(data["resting"], {"x": 0, "y": 0.5, "vx": 4, "vy": 0, "w": 1})
        self.assertEqual(data["fixed"], {"x": 0, "y": 0.2, "vx": 4, "vy": -5, "w": 0})

    def test_sphere_plane_friction_clamp_and_zero_restitution(self):
        data = self.run_node("""
            const P = require(process.argv[1]);
            const clamped = P.collideSpherePlane(
                {x: 0, y: 0.5, vx: 4, vy: -5, w: 1}, 0.5, 0, 0.5, 2
            );
            const negative = P.collideSpherePlane(
                {x: 0, y: 0.5, vx: 4, vy: -5, w: 1}, 0.5, 0, 0.5, -1
            );
            const inelastic = P.collideSpherePlane(
                {x: 0, y: 0.5, vx: 4, vy: -5, w: 1}, 0.5, 0, 0, 0
            );
            console.log(JSON.stringify({clamped, negative, inelastic}));
        """)
        self.assertEqual(data["clamped"], {"x": 0, "y": 0.5, "vx": 0, "vy": 2.5, "w": 1})
        self.assertEqual(data["negative"], {"x": 0, "y": 0.5, "vx": 8, "vy": 2.5, "w": 1})
        self.assertEqual(data["inelastic"], {"x": 0, "y": 0.5, "vx": 4, "vy": 0, "w": 1})

    def test_particle_sphere_impact_friction_and_purity(self):
        data = self.run_node("""
            const assert = require('node:assert/strict');
            const P = require(process.argv[1]);
            const input = Object.freeze({x: 0, y: 0.2, vx: 4, vy: -5, w: 1});
            const center = Object.freeze({x: 0, y: 0});
            const hit = P.collideParticleSphere(input, center, 0.3, 0.5, 0.5, 0.2);
            assert.equal(hit.y, 0.8);
            assert.equal(hit.vy, 2.5);
            assert.equal(hit.vx, 3.2);
            assert.notStrictEqual(hit, input);
            // n=(3/5,4/5), v.n=-16/5, tangent=(48/25,-36/25).
            // With friction=1/2 and restitution=1/2, v'=(48/25,14/25).
            const oblique = P.collideParticleSphere(
                Object.freeze({x: 0.48, y: 0.64, vx: 0, vy: -4, w: 1}),
                center, 0.3, 0.5, 0.5, 0.5
            );
            const clamped = P.collideParticleSphere(
                {x: 0, y: 0.2, vx: 4, vy: -5, w: 1}, center, 0.3, 0.5, 0.5, 2
            );
            console.log(JSON.stringify({hit, oblique, clamped, input, center}));
        """)
        self.assertEqual(data["hit"], {"x": 0, "y": 0.8, "vx": 3.2, "vy": 2.5, "w": 1})
        self.assertEqual(data["input"], {"x": 0, "y": 0.2, "vx": 4, "vy": -5, "w": 1})
        self.assertEqual(data["center"], {"x": 0, "y": 0})
        self.assertAlmostEqual(data["oblique"]["x"], 0.48, places=14)
        self.assertAlmostEqual(data["oblique"]["y"], 0.64, places=14)
        self.assertAlmostEqual(data["oblique"]["vx"], 48 / 25, places=14)
        self.assertAlmostEqual(data["oblique"]["vy"], 14 / 25, places=14)
        self.assertEqual(data["clamped"], {"x": 0, "y": 0.8, "vx": 0, "vy": 2.5, "w": 1})

    def test_particle_sphere_separating_fixed_and_coincident_boundaries(self):
        data = self.run_node("""
            const assert = require('node:assert/strict');
            const P = require(process.argv[1]);
            const center = Object.freeze({x: 0, y: 0});
            const above = P.collideParticleSphere(
                {x: 0, y: 0.9, vx: 4, vy: 2, w: 1}, center, 0.3, 0.5, 0.5, 0.2
            );
            const separating = P.collideParticleSphere(
                {x: 0, y: 0.2, vx: 4, vy: 2, w: 1}, center, 0.3, 0.5, 0.5, 0.2
            );
            const resting = P.collideParticleSphere(
                {x: 0, y: 0.8, vx: 4, vy: 0, w: 1}, center, 0.3, 0.5, 0.5, 0.2
            );
            const fixedInput = Object.freeze({x: 0, y: 0.2, vx: 4, vy: -5, w: 0});
            const fixed = P.collideParticleSphere(fixedInput, center, 0.3, 0.5, 0.5, 0.2);
            const coincidentInput = Object.freeze({x: 0, y: 0, vx: 4, vy: -5, w: 1});
            const coincident = P.collideParticleSphere(
                coincidentInput, center, 0.3, 0.5, 0.5, 0.2
            );
            assert.notStrictEqual(fixed, fixedInput);
            assert.notStrictEqual(coincident, coincidentInput);
            console.log(JSON.stringify({above, separating, resting, fixed, coincident}));
        """)
        self.assertEqual(data["above"], {"x": 0, "y": 0.9, "vx": 4, "vy": 2, "w": 1})
        self.assertEqual(data["separating"], {"x": 0, "y": 0.8, "vx": 4, "vy": 2, "w": 1})
        self.assertEqual(data["resting"], {"x": 0, "y": 0.8, "vx": 4, "vy": 0, "w": 1})
        self.assertEqual(data["fixed"], {"x": 0, "y": 0.2, "vx": 4, "vy": -5, "w": 0})
        self.assertEqual(data["coincident"], {"x": 0, "y": 0, "vx": 4, "vy": -5, "w": 1})

    def test_area_xpbd_compliance_direction_and_purity(self):
        data = self.run_node("""
            const assert = require('node:assert/strict');
            const P = require(process.argv[1]);
            const square = [
                {x: 0, y: 0, w: 1}, {x: 1, y: 0, w: 1},
                {x: 1, y: 1, w: 1}, {x: 0, y: 1, w: 1},
            ];
            const stretched = square.map((p, i) => i === 1 || i === 2 ? {...p, x: 2} : {...p});
            stretched.forEach(Object.freeze);
            Object.freeze(stretched);
            const areaHard = P.projectAreaXPBD(...stretched, 1, 0, 1 / 60, 0);
            const areaSoft = P.projectAreaXPBD(...stretched, 1, 1, 1 / 60, 0);
            const areaOf = (points) => Math.abs(points.reduce((sum, p, i) => {
              const q = points[(i + 1) % points.length];
              return sum + p.x*q.y - q.x*p.y;
            }, 0) / 2);
            // One gradient projection is not an exact quadratic area solve:
            // the brief's gradients give hard area 27/25, not exactly 1.
            assert.ok(Math.abs(areaOf(areaHard.particles) - 27 / 25) < 1e-12);
            assert.ok(Math.abs(areaOf(areaSoft.particles) - 1) > Math.abs(areaOf(areaHard.particles) - 1));
            areaHard.particles.forEach((p, i) => assert.notStrictEqual(p, stretched[i]));
            areaSoft.particles.forEach((p, i) => assert.notStrictEqual(p, stretched[i]));
            console.log(JSON.stringify({
                hardArea: areaOf(areaHard.particles), softArea: areaOf(areaSoft.particles),
                areaHard, areaSoft, stretched
            }));
        """)
        self.assertAlmostEqual(data["hardArea"], 27 / 25, places=12)
        self.assertEqual(data["areaHard"], {
            "particles": [
                {"x": 1 / 10, "y": 1 / 5, "w": 1},
                {"x": 19 / 10, "y": 1 / 5, "w": 1},
                {"x": 19 / 10, "y": 4 / 5, "w": 1},
                {"x": 1 / 10, "y": 4 / 5, "w": 1},
            ],
            "lambda": -1 / 5,
        })
        self.assertGreater(abs(data["softArea"] - 1), abs(data["hardArea"] - 1))
        self.assertEqual(data["stretched"], [
            {"x": 0, "y": 0, "w": 1}, {"x": 2, "y": 0, "w": 1},
            {"x": 2, "y": 1, "w": 1}, {"x": 0, "y": 1, "w": 1},
        ])

    def test_area_xpbd_zero_dt_returns_finite_clones_and_preserves_lambda(self):
        data = self.run_node("""
            const assert = require('node:assert/strict');
            const P = require(process.argv[1]);
            const stretched = Object.freeze([
                Object.freeze({x: 0, y: 0, w: 1}),
                Object.freeze({x: 2, y: 0, w: 1}),
                Object.freeze({x: 2, y: 1, w: 1}),
                Object.freeze({x: 0, y: 1, w: 1}),
            ]);
            const zeroArea = P.projectAreaXPBD(...stretched, 1, 1, 0, 0.25);
            assert.deepEqual(zeroArea.particles, stretched);
            zeroArea.particles.forEach((p, i) => {
              assert.notStrictEqual(p, stretched[i]);
              assert.ok(Number.isFinite(p.x));
              assert.ok(Number.isFinite(p.y));
            });
            assert.equal(zeroArea.lambda, 0.25);
            console.log(JSON.stringify({zeroArea, stretched}));
        """)
        self.assertEqual(data["zeroArea"], {
            "particles": [
                {"x": 0, "y": 0, "w": 1}, {"x": 2, "y": 0, "w": 1},
                {"x": 2, "y": 1, "w": 1}, {"x": 0, "y": 1, "w": 1},
            ],
            "lambda": 0.25,
        })
        self.assertEqual(data["stretched"], [
            {"x": 0, "y": 0, "w": 1}, {"x": 2, "y": 0, "w": 1},
            {"x": 2, "y": 1, "w": 1}, {"x": 0, "y": 1, "w": 1},
        ])

    def test_area_xpbd_clockwise_uses_original_winding(self):
        data = self.run_node("""
            const assert = require('node:assert/strict');
            const P = require(process.argv[1]);
            const stretched = [
                {x: 0, y: 0, w: 1}, {x: 2, y: 0, w: 1},
                {x: 2, y: 1, w: 1}, {x: 0, y: 1, w: 1},
            ];
            const clockwise = P.projectAreaXPBD(
                stretched[0], stretched[3], stretched[2], stretched[1],
                1, 0, 1 / 60, 0
            );
            const signedArea = (points) => points.reduce((sum, p, i) => {
              const q = points[(i + 1) % points.length];
              return sum + p.x*q.y - q.x*p.y;
            }, 0) / 2;
            assert.ok(Math.abs(signedArea(clockwise.particles) + 27 / 25) < 1e-12);
            // Keep the input's starting vertex: clockwise order is p0, p3, p2, p1.
            assert.deepEqual(clockwise.particles, [
                {x: 1 / 10, y: 1 / 5, w: 1}, {x: 1 / 10, y: 4 / 5, w: 1},
                {x: 19 / 10, y: 4 / 5, w: 1}, {x: 19 / 10, y: 1 / 5, w: 1},
            ]);
            // Clockwise gradients and C both flip sign, so delta_lambda is +1/5.
            assert.equal(clockwise.lambda, 1 / 5);
            console.log(JSON.stringify({
                clockwise, clockwiseSignedArea: signedArea(clockwise.particles)
            }));
        """)
        self.assertAlmostEqual(data["clockwiseSignedArea"], -27 / 25, places=12)
        self.assertEqual(data["clockwise"], {
            "particles": [
                {"x": 1 / 10, "y": 1 / 5, "w": 1},
                {"x": 1 / 10, "y": 4 / 5, "w": 1},
                {"x": 19 / 10, "y": 4 / 5, "w": 1},
                {"x": 19 / 10, "y": 1 / 5, "w": 1},
            ],
            "lambda": 1 / 5,
        })

    def test_area_xpbd_gradients_lambda_history_and_fixed_boundaries(self):
        data = self.run_node("""
            const assert = require('node:assert/strict');
            const P = require(process.argv[1]);
            const stretched = [
                {x: 0, y: 0, w: 1}, {x: 2, y: 0, w: 1},
                {x: 2, y: 1, w: 1}, {x: 0, y: 1, w: 1},
            ];
            // C=1; gradients=(-1/2,-1),(1/2,-1),(1/2,1),(-1/2,1); denom=5.
            // At dt=1/2, alpha_tilde=4 and delta_lambda=-1/9.
            const soft = P.projectAreaXPBD(...stretched, 1, 1, 0.5, 0);
            // Incoming lambda=1/4 gives delta_lambda=(-1-4*1/4)/9=-2/9.
            const history = P.projectAreaXPBD(...stretched, 1, 1, 0.5, 0.25);
            const oneFixed = P.projectAreaXPBD(
                {x: 0, y: 0, w: 0}, ...stretched.slice(1), 1, 1, 0.5, 0
            );
            const allFixed = P.projectAreaXPBD(
                ...stretched.map(p => ({...p, w: 0})), 1, 1, 0.5, 0.25
            );
            assert.equal(soft.lambda, -1 / 9);
            assert.ok(Math.abs(history.lambda - 1 / 36) < 1e-12);
            assert.equal(oneFixed.lambda, -4 / 31);
            assert.equal(allFixed.lambda, 0.25);
            console.log(JSON.stringify({soft, history, oneFixed, allFixed}));
        """)
        soft_particles = [
            {"x": 1 / 18, "y": 1 / 9, "w": 1},
            {"x": 35 / 18, "y": 1 / 9, "w": 1},
            {"x": 35 / 18, "y": 8 / 9, "w": 1},
            {"x": 1 / 18, "y": 8 / 9, "w": 1},
        ]
        self.assertEqual(data["soft"], {"particles": soft_particles, "lambda": -1 / 9})
        self.assertEqual(data["history"]["particles"], [
            {"x": 1 / 9, "y": 2 / 9, "w": 1},
            {"x": 17 / 9, "y": 2 / 9, "w": 1},
            {"x": 17 / 9, "y": 7 / 9, "w": 1},
            {"x": 1 / 9, "y": 7 / 9, "w": 1},
        ])
        self.assertAlmostEqual(data["history"]["lambda"], 1 / 36, places=14)
        one_fixed = data["oneFixed"]
        self.assertEqual(one_fixed["particles"][0], {"x": 0, "y": 0, "w": 0})
        for particle, expected in zip(one_fixed["particles"][1:], [
            {"x": 2 - 2 / 31, "y": 4 / 31, "w": 1},
            {"x": 2 - 2 / 31, "y": 27 / 31, "w": 1},
            {"x": 2 / 31, "y": 27 / 31, "w": 1},
        ]):
            self.assertAlmostEqual(particle["x"], expected["x"], places=14)
            self.assertAlmostEqual(particle["y"], expected["y"], places=14)
            self.assertEqual(particle["w"], 1)
        self.assertAlmostEqual(one_fixed["lambda"], -4 / 31, places=14)
        self.assertEqual(data["allFixed"], {
            "particles": [
                {"x": 0, "y": 0, "w": 0}, {"x": 2, "y": 0, "w": 0},
                {"x": 2, "y": 1, "w": 0}, {"x": 0, "y": 1, "w": 0},
            ],
            "lambda": 0.25,
        })

    def test_area_xpbd_degenerate_and_coincident_geometry_stays_finite(self):
        data = self.run_node("""
            const assert = require('node:assert/strict');
            const P = require(process.argv[1]);
            const coincidentPoint = Object.freeze({x: 4, y: 5, w: 1});
            const coincident = P.projectAreaXPBD(
                coincidentPoint, coincidentPoint, coincidentPoint, coincidentPoint,
                1, 1, 0.5, 0.25
            );
            coincident.particles.forEach(p => assert.notStrictEqual(p, coincidentPoint));
            const collinear = P.projectAreaXPBD(
                {x: 0, y: 0, w: 1}, {x: 1, y: 0, w: 1},
                {x: 2, y: 0, w: 1}, {x: 3, y: 0, w: 1},
                1, 0, 0.5, 0
            );
            // Zero area gives C=-1, denom=4, delta_lambda=1/4.
            assert.equal(collinear.lambda, 0.25);
            console.log(JSON.stringify({coincident, collinear}));
        """)
        self.assertEqual(data["coincident"], {
            "particles": [{"x": 4, "y": 5, "w": 1}] * 4,
            "lambda": 0.25,
        })
        self.assertEqual(data["collinear"], {
            "particles": [
                {"x": 0, "y": 0.25, "w": 1}, {"x": 1, "y": -0.25, "w": 1},
                {"x": 2, "y": -0.25, "w": 1}, {"x": 3, "y": 0.25, "w": 1},
            ],
            "lambda": 0.25,
        })
        self.assertTrue(all(isinstance(value, (int, float)) and abs(value) < float("inf")
                            for value in self.flatten(data)))

    def test_umd_exports_same_core_in_node_and_browser(self):
        data = self.run_node("""
            const assert = require('node:assert/strict');
            const fs = require('node:fs');
            const vm = require('node:vm');
            const nodeCore = require(process.argv[1]);
            const browserWindow = {};
            vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), {
                window: browserWindow
            });
            assert.ok(browserWindow.RTPhysics);
            assert.equal(typeof browserWindow.RTPhysics.integrateProjectile, 'function');
            assert.equal(browserWindow.RTPhysics.vec2.dot({x: 3, y: 4}, {x: 2, y: -1}), 2);
            assert.equal(browserWindow.RTPhysics.stableSpringStep(100, 4), 0.4);
            assert.ok(Math.abs(nodeCore.integrateProjectile(
                {x: 0, y: 0, vx: 10, vy: 20}, {x: 0, y: -9.81}, 1, 'exact'
            ).y - 15.095) < 1e-12);
            console.log(JSON.stringify({
                nodeExportType: typeof nodeCore,
                browserExportType: typeof browserWindow.RTPhysics,
                vectorKeys: Object.keys(nodeCore.vec2).sort()
            }));
        """)
        self.assertEqual(data["nodeExportType"], "object")
        self.assertEqual(data["browserExportType"], "object")
        self.assertEqual(data["vectorKeys"], ["add", "dot", "length", "normalize", "scale", "sub"])

    @staticmethod
    def flatten(value):
        if isinstance(value, dict):
            for child in value.values():
                yield from PhysicsCoreTests.flatten(child)
        elif isinstance(value, list):
            for child in value:
                yield from PhysicsCoreTests.flatten(child)
        else:
            yield value


if __name__ == "__main__":
    unittest.main()
