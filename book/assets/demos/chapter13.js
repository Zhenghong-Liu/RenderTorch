(() => {
  const mount = document.getElementById("texture-demo");
  if (!mount) return;

  mount.insertAdjacentHTML(
    "beforeend",
    `<div class="demo-toolbar">
      <label>过滤模式<select id="texture-filter" data-filter><option value="nearest">nearest</option><option value="bilinear">bilinear</option></select></label>
      <label>远端顶点 C 的 w <input id="texture-far-w" type="range" data-depth min="1" max="8" step="0.1" value="4"><output for="texture-far-w">4.0</output></label>
      <button type="button" data-reset>重置</button>
    </div>
    <div class="demo-grid">
      <figure><canvas id="texture-linear-canvas" width="332" height="292" role="img" aria-label="屏幕线性 UV 错误插值的棋盘纹理三角形"></canvas><figcaption>屏幕线性 UV（错误）</figcaption></figure>
      <figure><canvas id="texture-correct-canvas" width="332" height="292" role="img" aria-label="透视矫正 UV 正确插值的棋盘纹理三角形"></canvas><figcaption>透视矫正 UV（正确）</figcaption></figure>
    </div>
    <p class="demo-readout" data-readout aria-live="polite" aria-atomic="true"></p>`
  );

  const linearCanvas = mount.querySelector("#texture-linear-canvas");
  const correctCanvas = mount.querySelector("#texture-correct-canvas");
  const linearContext = linearCanvas.getContext("2d");
  const correctContext = correctCanvas.getContext("2d");
  const filterSelect = mount.querySelector("[data-filter]");
  const depthInput = mount.querySelector("[data-depth]");
  const depthOutput = depthInput.nextElementSibling;
  const readout = mount.querySelector("[data-readout]");
  const resetButton = mount.querySelector("[data-reset]");
  const defaults = { filter: "nearest", farW: 4 };
  const fmt = (value) => {
    if (!Number.isFinite(value)) return "—";
    if (window.RTBook && typeof window.RTBook.formatNumber === "function") return window.RTBook.formatNumber(value, 3);
    return value.toFixed(3);
  };
  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
  const vertices = [
    { x: 48, y: 236, u: 0, v: 0, w: 1 },
    { x: 294, y: 236, u: 1, v: 0, w: 1 },
    { x: 171, y: 52, u: 1, v: 1, w: defaults.farW },
  ];
  const samplePoint = {
    x: (vertices[0].x + vertices[1].x + vertices[2].x) / 3,
    y: (vertices[0].y + vertices[1].y + vertices[2].y) / 3,
  };
  const light = [246, 243, 235];
  const dark = [25, 67, 73];
  const background = [16, 36, 44];
  const texture = Array.from({ length: 8 }, (_, row) =>
    Array.from({ length: 8 }, (_, column) => {
      const u = (column + 0.5) / 8;
      const v = (row + 0.5) / 8;
      const checker = ((Math.floor(u * 8) + Math.floor(v * 8)) % 2) === 0;
      return checker ? light : dark;
    })
  );

  if (!linearContext || !correctContext) {
    readout.textContent = "当前浏览器不支持 Canvas 2D，无法绘制纹理演示。";
    return;
  }

  function edge(a, b, p) {
    return (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);
  }

  function barycentric(p) {
    const area = edge(vertices[0], vertices[1], vertices[2]);
    if (!Number.isFinite(area) || Math.abs(area) < 1e-12) return null;
    const lambda = [
      edge(vertices[1], vertices[2], p) / area,
      edge(vertices[2], vertices[0], p) / area,
      edge(vertices[0], vertices[1], p) / area,
    ];
    if (!lambda.every(Number.isFinite)) return null;
    return lambda;
  }

  function interpolateUV(lambda, farW, perspectiveCorrect) {
    if (!lambda || !lambda.every((value) => Number.isFinite(value) && value >= 0)) return null;
    const weights = [1, 1, farW];
    if (!weights.every((value) => Number.isFinite(value) && value > 0)) return null;
    const linearU = lambda.reduce((sum, value, index) => sum + value * vertices[index].u, 0);
    const linearV = lambda.reduce((sum, value, index) => sum + value * vertices[index].v, 0);
    if (!perspectiveCorrect) return { u: linearU, v: linearV, invW: null };
    const invW = lambda.reduce((sum, value, index) => sum + value / weights[index], 0);
    if (!Number.isFinite(invW) || invW <= 0) return null;
    const u = lambda.reduce((sum, value, index) => sum + value * vertices[index].u / weights[index], 0) / invW;
    const v = lambda.reduce((sum, value, index) => sum + value * vertices[index].v / weights[index], 0) / invW;
    if (![u, v].every(Number.isFinite)) return null;
    return { u, v, invW };
  }

  function sampleTexture(u, v, filter) {
    if (![u, v].every(Number.isFinite)) return null;
    const sampleU = clamp(u, 0, 1);
    const sampleV = clamp(v, 0, 1);
    if (filter === "nearest") {
      const x = Math.min(7, Math.floor(sampleU * 8));
      const y = Math.min(7, Math.floor(sampleV * 8));
      return texture[y][x];
    }
    const x = sampleU * 8 - 0.5;
    const y = sampleV * 8 - 0.5;
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const fx = x - x0;
    const fy = y - y0;
    const texel = (column, row) => texture[clamp(row, 0, 7)][clamp(column, 0, 7)];
    const c00 = texel(x0, y0);
    const c10 = texel(x0 + 1, y0);
    const c01 = texel(x0, y0 + 1);
    const c11 = texel(x0 + 1, y0 + 1);
    return c00.map((value, channel) =>
      Math.round(
        value * (1 - fx) * (1 - fy) +
          c10[channel] * fx * (1 - fy) +
          c01[channel] * (1 - fx) * fy +
          c11[channel] * fx * fy
      )
    );
  }

  function drawTriangle(context, farW, filter, perspectiveCorrect) {
    const image = context.createImageData(context.canvas.width, context.canvas.height);
    const pixels = image.data;
    for (let y = 0; y < context.canvas.height; y += 1) {
      for (let x = 0; x < context.canvas.width; x += 1) {
        const offset = (y * context.canvas.width + x) * 4;
        pixels[offset] = background[0];
        pixels[offset + 1] = background[1];
        pixels[offset + 2] = background[2];
        pixels[offset + 3] = 255;
        const lambda = barycentric({ x: x + 0.5, y: y + 0.5 });
        if (!lambda || lambda.some((value) => value < 0)) continue;
        const uv = interpolateUV(lambda, farW, perspectiveCorrect);
        if (!uv) continue;
        const color = sampleTexture(uv.u, uv.v, filter);
        if (!color) continue;
        pixels[offset] = color[0];
        pixels[offset + 1] = color[1];
        pixels[offset + 2] = color[2];
      }
    }
    context.putImageData(image, 0, 0);

    context.fillStyle = "#9db6b8";
    context.font = "12px system-ui";
    context.fillText("UV: A(0,0) · B(1,0) · C(1,1)", 14, 24);
    context.strokeStyle = "rgba(246,243,235,.72)";
    context.lineWidth = 1.5;
    context.beginPath();
    context.moveTo(vertices[0].x, vertices[0].y);
    context.lineTo(vertices[1].x, vertices[1].y);
    context.lineTo(vertices[2].x, vertices[2].y);
    context.closePath();
    context.stroke();
    context.lineWidth = 1;

    context.font = "bold 12px system-ui";
    context.fillStyle = "#ffd48a";
    vertices.forEach((vertex, index) => {
      context.beginPath();
      context.arc(vertex.x, vertex.y, 3.5, 0, Math.PI * 2);
      context.fill();
      context.fillText(["A", "B", "C"][index], vertex.x + 7, vertex.y - 7);
    });

    context.strokeStyle = "#ffd48a";
    context.lineWidth = 2;
    context.beginPath();
    context.moveTo(samplePoint.x - 9, samplePoint.y);
    context.lineTo(samplePoint.x + 9, samplePoint.y);
    context.moveTo(samplePoint.x, samplePoint.y - 9);
    context.lineTo(samplePoint.x, samplePoint.y + 9);
    context.stroke();
    context.lineWidth = 1;
    context.fillStyle = "#9db6b8";
    context.font = "12px system-ui";
    context.fillText(`p=(${fmt(samplePoint.x)}, ${fmt(samplePoint.y)})`, 14, 276);
    context.textAlign = "right";
    context.fillText(`w=(1, 1, ${fmt(farW)})`, 318, 276);
    context.textAlign = "left";
  }

  function draw() {
    depthOutput.value = fmt(Number(depthInput.value));
    const farW = Number(depthInput.value);
    const filter = filterSelect.value === "bilinear" ? "bilinear" : "nearest";
    drawTriangle(linearContext, farW, filter, false);
    drawTriangle(correctContext, farW, filter, true);
    const lambda = barycentric(samplePoint);
    const linearUV = interpolateUV(lambda, farW, false);
    const correctUV = interpolateUV(lambda, farW, true);
    const lambdaText = lambda
      ? lambda.map((value, index) => `λ${index}=${fmt(value)}`).join(" · ")
      : "λ=—";
    if (!linearUV || !correctUV) {
      readout.textContent = `样本 p=(${fmt(samplePoint.x)}, ${fmt(samplePoint.y)}) · ${lambdaText} · w 或 1/w 非正/非有限，已安全跳过纹理采样 · 过滤=${filter}`;
      return;
    }
    readout.textContent =
      `样本 p=(${fmt(samplePoint.x)}, ${fmt(samplePoint.y)}) · ${lambdaText} · ` +
      `线性 UV=(${fmt(linearUV.u)}, ${fmt(linearUV.v)}) · ` +
      `透视矫正 UV=(${fmt(correctUV.u)}, ${fmt(correctUV.v)}) · ` +
      `1/w=${fmt(correctUV.invW)} · 过滤=${filter}`;
  }

  depthInput.addEventListener("input", draw);
  filterSelect.addEventListener("change", draw);
  resetButton.addEventListener("click", () => {
    depthInput.value = String(defaults.farW);
    filterSelect.value = defaults.filter;
    draw();
  });

  draw();
})();
