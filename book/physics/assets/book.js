(() => {
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  window.RTBook = {
    clamp(value, min, max) {
      return Math.min(max, Math.max(min, value));
    },
    lerp(a, b, t) {
      return a + (b - a) * t;
    },
    normalize2(x, y) {
      const length = Math.hypot(x, y) || 1;
      return { x: x / length, y: y / length };
    },
    setupCanvas(canvas) {
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      const width = Math.max(1, Math.round(rect.width * ratio));
      const height = Math.max(1, Math.round(rect.height * ratio));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      const ctx = canvas.getContext("2d");
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      return { ctx, width: rect.width, height: rect.height };
    },
    formatNumber(value, digits = 3) {
      return Number.isFinite(value) ? value.toFixed(digits) : "—";
    },
    prefersReducedMotion() {
      return reducedMotion.matches;
    },
  };

  document.addEventListener("DOMContentLoaded", () => {
    if (window.renderMathInElement) {
      window.renderMathInElement(document.body, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "\\[", right: "\\]", display: true },
          { left: "\\(", right: "\\)", display: false },
          { left: "$", right: "$", display: false },
        ],
        throwOnError: false,
        errorCallback(message, error) {
          console.error("KaTeX formula error:", message, error);
        },
      });
    }

    const toggle = document.querySelector("[data-menu-toggle]");
    const sidebar = document.querySelector("[data-sidebar]");

    const closeMenu = () => {
      if (!sidebar || !toggle) return;
      const restoreFocus = sidebar.contains(document.activeElement);
      sidebar.dataset.open = "false";
      toggle.setAttribute("aria-expanded", "false");
      if (restoreFocus) toggle.focus();
    };

    if (toggle && sidebar) {
      toggle.addEventListener("click", () => {
        const open = sidebar.dataset.open !== "true";
        sidebar.dataset.open = String(open);
        toggle.setAttribute("aria-expanded", String(open));
        if (open) sidebar.querySelector("a")?.focus();
      });
      sidebar.addEventListener("click", (event) => {
        if (event.target.closest("a")) closeMenu();
      });
      document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") closeMenu();
      });
    }

    document.querySelectorAll("[data-copy]").forEach((button) => {
      button.addEventListener("click", async () => {
        const card = button.closest(".code-card");
        const code = card?.querySelector("code");
        if (!code) return;
        try {
          await navigator.clipboard.writeText(code.textContent);
          button.textContent = "已复制";
        } catch {
          button.textContent = "复制失败";
        }
        window.setTimeout(() => {
          button.textContent = "复制代码";
        }, 1400);
      });
    });
  });
})();
