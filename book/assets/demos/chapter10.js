(() => {
  const mount = document.getElementById("raster-demo");
  if (!mount) return;

  mount.insertAdjacentHTML(
    "beforeend",
    `<div class="demo-toolbar">
      <label>操作点<select data-point><option value="0">顶点 A</option><option value="1">顶点 B</option><option value="2">顶点 C</option><option value="3">探针 p</option></select></label>
      <label>x <input type="range" data-coordinate="x" min="0" max="10" step="0.1" value="1"><output>1.0</output></label>
      <label>y <input type="range" data-coordinate="y" min="0" max="7" step="0.1" value="1"><output>1.0</output></label>
      <button type="button" data-reset>重置</button>
    </div>
    <canvas id="raster-canvas" width="720" height="360" role="img" aria-label="可拖动的三角形与像素中心覆盖测试"></canvas>
    <p class="demo-readout" data-readout aria-live="polite"></p>`
  );

  const canvas = mount.querySelector("#raster-canvas");
  const context = canvas.getContext("2d");
  const pointSelect = mount.querySelector("[data-point]");
  const inputs = [...mount.querySelectorAll("[data-coordinate]")];
  const readout = mount.querySelector("[data-readout]");
  const grid = { left: 35, top: 10, columns: 10, rows: 7, cell: 46 };
  const defaults = [[1, 1], [8, 1], [1, 6], [3.5, 2.5]];
  const points = defaults.map((point) => ({ x: point[0], y: point[1] }));
  let selected = 0;
  let dragging = null;
  const fmt = (value) => window.RTBook.formatNumber(value, 3);

  function edge(a, b, p) {
    return (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);
  }

  function barycentric() {
    const area = edge(points[0], points[1], points[2]);
    if (Math.abs(area) < 1e-12) return { area, l0: NaN, l1: NaN, l2: NaN, covered: false, degenerate: true };
    const l0 = edge(points[1], points[2], points[3]) / area;
    const l1 = edge(points[2], points[0], points[3]) / area;
    const l2 = 1 - l0 - l1;
    const covered = Number.isFinite(l0) && Number.isFinite(l1) && Number.isFinite(l2) && l0 >= 0 && l1 >= 0 && l2 >= 0;
    return { area, l0, l1, l2, covered, degenerate: false };
  }

  function toScreen(point) {
    return {
      x: grid.left + point.x * grid.cell,
      y: grid.top + (grid.rows - point.y) * grid.cell,
    };
  }

  function toWorld(x, y) {
    return {
      x: window.RTBook.clamp((x - grid.left) / grid.cell, 0, grid.columns),
      y: window.RTBook.clamp(grid.rows - (y - grid.top) / grid.cell, 0, grid.rows),
    };
  }

  function syncControls() {
    const point = points[selected];
    inputs.forEach((input) => {
      input.value = String(point[input.dataset.coordinate]);
      input.nextElementSibling.value = fmt(point[input.dataset.coordinate]);
    });
  }

  function drawMarker(point, label, color, radius = 6) {
    const screen = toScreen(point);
    context.fillStyle = color;
    context.beginPath();
    context.arc(screen.x, screen.y, radius, 0, Math.PI * 2);
    context.fill();
    context.font = "bold 14px system-ui";
    context.fillText(label, screen.x + 9, screen.y - 8);
  }

  function draw() {
    const result = barycentric();
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#10242c";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#9db6b8";
    context.font = "14px system-ui";
    context.fillText("像素中心采样 · 每格 1 × 1", grid.left, 30);
    context.fillText("拖 A / B / C / p", 545, 30);

    for (let y = 0; y < grid.rows; y += 1) {
      for (let x = 0; x < grid.columns; x += 1) {
        const center = { x: x + 0.5, y: y + 0.5 };
        if (result.degenerate) continue;
        const l0 = edge(points[1], points[2], center) / result.area;
        const l1 = edge(points[2], points[0], center) / result.area;
        const l2 = 1 - l0 - l1;
        const covered = l0 >= 0 && l1 >= 0 && l2 >= 0;
        context.fillStyle = covered ? "rgba(88,199,189,.17)" : "rgba(157,182,184,.04)";
        context.fillRect(grid.left + x * grid.cell, grid.top + (grid.rows - 1 - y) * grid.cell, grid.cell, grid.cell);
      }
    }

    context.strokeStyle = "rgba(255,255,255,.16)";
    context.lineWidth = 1;
    for (let x = 0; x <= grid.columns; x += 1) {
      context.beginPath();
      context.moveTo(grid.left + x * grid.cell, grid.top);
      context.lineTo(grid.left + x * grid.cell, grid.top + grid.rows * grid.cell);
      context.stroke();
    }
    for (let y = 0; y <= grid.rows; y += 1) {
      context.beginPath();
      context.moveTo(grid.left, grid.top + y * grid.cell);
      context.lineTo(grid.left + grid.columns * grid.cell, grid.top + y * grid.cell);
      context.stroke();
    }

    const triangle = points.slice(0, 3).map(toScreen);
    context.strokeStyle = result.degenerate ? "#ef9a67" : "#58c7bd";
    context.lineWidth = 3;
    context.beginPath();
    context.moveTo(triangle[0].x, triangle[0].y);
    context.lineTo(triangle[1].x, triangle[1].y);
    context.lineTo(triangle[2].x, triangle[2].y);
    context.closePath();
    context.stroke();
    context.lineWidth = 1;

    drawMarker(points[0], "A", "#ef9a67");
    drawMarker(points[1], "B", "#ef9a67");
    drawMarker(points[2], "C", "#ef9a67");
    const probe = toScreen(points[3]);
    context.strokeStyle = result.covered ? "#58c7bd" : "#ffd48a";
    context.lineWidth = 3;
    context.beginPath();
    context.moveTo(probe.x - 10, probe.y);
    context.lineTo(probe.x + 10, probe.y);
    context.moveTo(probe.x, probe.y - 10);
    context.lineTo(probe.x, probe.y + 10);
    context.stroke();
    context.lineWidth = 1;
    drawMarker(points[3], "p", result.covered ? "#58c7bd" : "#ffd48a", 4);

    context.fillStyle = "#9db6b8";
    context.font = "13px system-ui";
    context.fillText(`p=(${fmt(points[3].x)}, ${fmt(points[3].y)})`, 545, 74);
    context.fillText(result.covered ? "p 在三角形内" : "p 在三角形外", 545, 98);
    context.fillStyle = result.degenerate ? "#ef9a67" : "#58c7bd";
    context.fillText(result.degenerate ? "面积过小：跳过光栅化" : `有符号面积 A=${fmt(result.area)}`, 545, 126);

    if (result.degenerate) {
      readout.textContent = "面积过小（|A| < 1e-12）：三角形退化，不计算重心坐标，也不写入像素。";
    } else {
      readout.textContent =
        `A=${fmt(result.area)} · λ0=${fmt(result.l0)} · λ1=${fmt(result.l1)} · λ2=${fmt(result.l2)} · ` +
        `p=(${fmt(points[3].x)}, ${fmt(points[3].y)}) · ${result.covered ? "覆盖" : "未覆盖"}`;
    }
  }

  function pointerPosition(event) {
    const bounds = canvas.getBoundingClientRect();
    return {
      x: ((event.clientX - bounds.left) / bounds.width) * canvas.width,
      y: ((event.clientY - bounds.top) / bounds.height) * canvas.height,
    };
  }

  function nearestPoint(position) {
    let best = null;
    let distance = 24;
    points.forEach((point, index) => {
      const screen = toScreen(point);
      const current = Math.hypot(screen.x - position.x, screen.y - position.y);
      if (current < distance) {
        best = index;
        distance = current;
      }
    });
    return best;
  }

  pointSelect.addEventListener("change", () => {
    selected = Number(pointSelect.value);
    syncControls();
    draw();
  });
  inputs.forEach((input) => {
    input.addEventListener("input", () => {
      points[selected][input.dataset.coordinate] = Number(input.value);
      input.nextElementSibling.value = fmt(points[selected][input.dataset.coordinate]);
      draw();
    });
  });
  canvas.addEventListener("pointerdown", (event) => {
    const index = nearestPoint(pointerPosition(event));
    if (index === null) return;
    dragging = index;
    selected = index;
    pointSelect.value = String(index);
    canvas.setPointerCapture(event.pointerId);
    syncControls();
    draw();
  });
  canvas.addEventListener("pointermove", (event) => {
    if (dragging === null) return;
    points[dragging] = toWorld(pointerPosition(event).x, pointerPosition(event).y);
    syncControls();
    draw();
  });
  const stopDragging = (event) => {
    if (dragging === null) return;
    dragging = null;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  };
  canvas.addEventListener("pointerup", stopDragging);
  canvas.addEventListener("pointercancel", stopDragging);
  mount.querySelector("[data-reset]").addEventListener("click", () => {
    defaults.forEach((point, index) => {
      points[index].x = point[0];
      points[index].y = point[1];
    });
    selected = 0;
    dragging = null;
    pointSelect.value = "0";
    syncControls();
    draw();
  });

  syncControls();
  draw();
})();
