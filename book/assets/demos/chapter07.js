(() => {
  const mount = document.getElementById("curve-demo");
  if (!mount) return;

  mount.insertAdjacentHTML(
    "beforeend",
    `<div class="demo-toolbar">
      <label>控制点 <select data-point-select><option value="0">P0</option><option value="1">P1</option><option value="2">P2</option><option value="3">P3</option></select></label>
      <label>x <input type="range" data-point-x min="-1" max="5" step="0.1" value="0"><output>0</output></label>
      <label>y <input type="range" data-point-y min="-1" max="3" step="0.1" value="0"><output>0</output></label>
      <label>t <input type="range" data-t min="0" max="1" step="0.01" value="0.5"><output>0.50</output></label>
      <button type="button" data-reset>重置</button>
    </div>
    <canvas id="curve-canvas" width="720" height="360" role="img" aria-label="可拖动四个控制点的三次贝塞尔曲线与 de Casteljau 中间级"></canvas>
    <p class="demo-readout" data-readout aria-live="polite"></p>`
  );

  const canvas = mount.querySelector("#curve-canvas");
  const context = canvas.getContext("2d");
  const readout = mount.querySelector("[data-readout]");
  const pointSelect = mount.querySelector("[data-point-select]");
  const xInput = mount.querySelector("[data-point-x]");
  const yInput = mount.querySelector("[data-point-y]");
  const tInput = mount.querySelector("[data-t]");
  const reset = mount.querySelector("[data-reset]");
  const defaults = [{ x: 0, y: 0 }, { x: 1, y: 2 }, { x: 3, y: 2 }, { x: 4, y: 0 }];
  let points = defaults.map((point) => ({ ...point }));
  let dragging = -1;
  const bounds = { minX: -0.8, maxX: 4.8, minY: -0.6, maxY: 2.6 };
  const plot = { left: 55, top: 35, width: 610, height: 280 };

  const lerpPoint = (a, b, t) => ({ x: window.RTBook.lerp(a.x, b.x, t), y: window.RTBook.lerp(a.y, b.y, t) });
  const subtract = (a, b) => ({ x: a.x - b.x, y: a.y - b.y });
  const toScreen = (point) => ({
    x: plot.left + ((point.x - bounds.minX) / (bounds.maxX - bounds.minX)) * plot.width,
    y: plot.top + plot.height - ((point.y - bounds.minY) / (bounds.maxY - bounds.minY)) * plot.height,
  });
  const toWorld = (screen) => ({
    x: bounds.minX + ((screen.x - plot.left) / plot.width) * (bounds.maxX - bounds.minX),
    y: bounds.minY + ((plot.top + plot.height - screen.y) / plot.height) * (bounds.maxY - bounds.minY),
  });

  function deCasteljau(t) {
    const q = points.slice(0, -1).map((point, index) => lerpPoint(point, points[index + 1], t));
    const r = [lerpPoint(q[0], q[1], t), lerpPoint(q[1], q[2], t)];
    return { q, r, point: lerpPoint(r[0], r[1], t) };
  }

  function derivative(t) {
    const weights = [(1 - t) ** 2, 2 * t * (1 - t), t ** 2];
    return points.slice(0, -1).reduce((sum, point, index) => {
      const difference = subtract(points[index + 1], point);
      return { x: sum.x + 3 * weights[index] * difference.x, y: sum.y + 3 * weights[index] * difference.y };
    }, { x: 0, y: 0 });
  }

  function arrow(start, end, color) {
    const angle = Math.atan2(end.y - start.y, end.x - start.x);
    context.strokeStyle = color;
    context.fillStyle = color;
    context.lineWidth = 3;
    context.beginPath();
    context.moveTo(start.x, start.y);
    context.lineTo(end.x, end.y);
    context.stroke();
    context.beginPath();
    context.moveTo(end.x, end.y);
    context.lineTo(end.x - 10 * Math.cos(angle - 0.45), end.y - 10 * Math.sin(angle - 0.45));
    context.lineTo(end.x - 10 * Math.cos(angle + 0.45), end.y - 10 * Math.sin(angle + 0.45));
    context.closePath();
    context.fill();
    context.lineWidth = 1;
  }

  function drawPoint(point, color, radius, label) {
    const screen = toScreen(point);
    context.beginPath();
    context.arc(screen.x, screen.y, radius, 0, Math.PI * 2);
    context.fillStyle = color;
    context.fill();
    context.fillStyle = "#edf5f3";
    context.font = "14px system-ui";
    context.fillText(label, screen.x + 8, screen.y - 8);
  }

  function syncControls() {
    const point = points[Number(pointSelect.value)];
    xInput.value = String(point.x);
    yInput.value = String(point.y);
    xInput.nextElementSibling.value = window.RTBook.formatNumber(point.x, 1);
    yInput.nextElementSibling.value = window.RTBook.formatNumber(point.y, 1);
    tInput.nextElementSibling.value = Number(tInput.value).toFixed(2);
  }

  function draw() {
    const t = Number(tInput.value);
    const levels = deCasteljau(t);
    const tangent = derivative(t);
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#10242c";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.strokeStyle = "rgba(255,255,255,.12)";
    context.strokeRect(plot.left, plot.top, plot.width, plot.height);

    context.beginPath();
    points.forEach((point, index) => {
      const screen = toScreen(point);
      if (index === 0) context.moveTo(screen.x, screen.y);
      else context.lineTo(screen.x, screen.y);
    });
    context.strokeStyle = "#789096";
    context.lineWidth = 2;
    context.setLineDash([7, 6]);
    context.stroke();
    context.setLineDash([]);

    context.beginPath();
    for (let step = 0; step <= 120; step += 1) {
      const point = toScreen(deCasteljau(step / 120).point);
      if (step === 0) context.moveTo(point.x, point.y);
      else context.lineTo(point.x, point.y);
    }
    context.strokeStyle = "#58c7bd";
    context.lineWidth = 4;
    context.stroke();

    const levelGroups = [[...points], levels.q, levels.r, [levels.point]];
    const levelColors = ["#789096", "#ef9a67", "#ffd48a", "#edf5f3"];
    levelGroups.forEach((group, groupIndex) => {
      for (let index = 0; index < group.length - 1; index += 1) {
        const a = toScreen(group[index]);
        const b = toScreen(group[index + 1]);
        context.beginPath();
        context.moveTo(a.x, a.y);
        context.lineTo(b.x, b.y);
        context.strokeStyle = levelColors[groupIndex];
        context.lineWidth = groupIndex === 0 ? 1.5 : 2.5;
        context.stroke();
      }
      group.forEach((point, index) => drawPoint(point, levelColors[groupIndex], groupIndex === 0 ? 7 : 5, groupIndex === 0 ? `P${index}` : groupIndex === 3 ? "B(t)" : ""));
    });

    const curvePoint = toScreen(levels.point);
    const tangentEnd = toScreen({ x: levels.point.x + tangent.x * 0.25, y: levels.point.y + tangent.y * 0.25 });
    arrow(curvePoint, tangentEnd, "#ffd48a");
    context.fillStyle = "#ffd48a";
    context.fillText("B'(t)", tangentEnd.x + 7, tangentEnd.y - 5);

    const fmt = (point) => `(${window.RTBook.formatNumber(point.x, 3)}, ${window.RTBook.formatNumber(point.y, 3)})`;
    readout.textContent =
      `t=${t.toFixed(2)} · B(t)=${fmt(levels.point)} · ` +
      `Q=[${levels.q.map(fmt).join(", ")}] · R=[${levels.r.map(fmt).join(", ")}] · ` +
      `B'(t)=${fmt(tangent)}`;
  }

  function updatePoint(axis, value) {
    points[Number(pointSelect.value)][axis] = window.RTBook.clamp(value, axis === "x" ? -1 : -1, axis === "x" ? 5 : 3);
    syncControls();
    draw();
  }

  pointSelect.addEventListener("change", () => { syncControls(); draw(); });
  xInput.addEventListener("input", () => updatePoint("x", Number(xInput.value)));
  yInput.addEventListener("input", () => updatePoint("y", Number(yInput.value)));
  tInput.addEventListener("input", () => { syncControls(); draw(); });
  reset.addEventListener("click", () => {
    points = defaults.map((point) => ({ ...point }));
    pointSelect.value = "0";
    tInput.value = "0.5";
    syncControls();
    draw();
  });

  function pointerPosition(event) {
    const rect = canvas.getBoundingClientRect();
    return { x: ((event.clientX - rect.left) / rect.width) * canvas.width, y: ((event.clientY - rect.top) / rect.height) * canvas.height };
  }

  canvas.addEventListener("pointerdown", (event) => {
    const pointer = pointerPosition(event);
    let nearest = -1;
    let distance = 22;
    points.forEach((point, index) => {
      const screen = toScreen(point);
      const candidate = Math.hypot(screen.x - pointer.x, screen.y - pointer.y);
      if (candidate < distance) { nearest = index; distance = candidate; }
    });
    if (nearest < 0) return;
    dragging = nearest;
    pointSelect.value = String(nearest);
    canvas.setPointerCapture(event.pointerId);
    syncControls();
  });
  canvas.addEventListener("pointermove", (event) => {
    if (dragging < 0) return;
    const world = toWorld(pointerPosition(event));
    points[dragging].x = window.RTBook.clamp(world.x, -1, 5);
    points[dragging].y = window.RTBook.clamp(world.y, -1, 3);
    syncControls();
    draw();
  });
  const stopDragging = () => { dragging = -1; };
  canvas.addEventListener("pointerup", stopDragging);
  canvas.addEventListener("pointercancel", stopDragging);

  syncControls();
  draw();
})();
