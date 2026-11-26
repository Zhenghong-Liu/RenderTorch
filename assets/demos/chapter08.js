(() => {
  const mount = document.getElementById("surface-demo");
  if (!mount) return;

  mount.insertAdjacentHTML(
    "beforeend",
    `<div class="demo-toolbar">
      <label>u <input type="range" data-u min="0" max="1" step="0.01" value="0.5"><output>0.50</output></label>
      <label>v <input type="range" data-v min="0" max="1" step="0.01" value="0.5"><output>0.50</output></label>
      <label><input type="radio" name="surface-mode" value="raw" checked> 原始网格</label>
      <label><input type="radio" name="surface-mode" value="subdivided"> 一次细分</label>
      <label><input type="radio" name="surface-mode" value="both"> 叠加显示</label>
      <button type="button" data-reset>重置</button>
    </div>
    <canvas id="surface-canvas" width="720" height="360" role="img" aria-label="双线性曲面原始网格与一次边中点细分网格"></canvas>
    <p class="demo-readout" data-readout aria-live="polite"></p>`
  );

  const canvas = mount.querySelector("#surface-canvas");
  const context = canvas.getContext("2d");
  const readout = mount.querySelector("[data-readout]");
  const uInput = mount.querySelector("[data-u]");
  const vInput = mount.querySelector("[data-v]");
  const modes = [...mount.querySelectorAll('[name="surface-mode"]')];
  const reset = mount.querySelector("[data-reset]");
  const control = {
    p00: [0, 0, 0], p10: [2, 0, 0],
    p01: [0, 1, 0], p11: [2, 1, 1],
  };
  const lerp = (a, b, t) => a.map((value, index) => window.RTBook.lerp(value, b[index], t));
  const add = (a, b) => a.map((value, index) => value + b[index]);
  const scale = (a, factor) => a.map((value) => value * factor);
  const subtract = (a, b) => a.map((value, index) => value - b[index]);
  const dot = (a, b) => a.reduce((sum, value, index) => sum + value * b[index], 0);
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const normalize = (vector) => {
    const length = Math.hypot(...vector);
    return length > 1e-12 ? scale(vector, 1 / length) : [0, 0, 0];
  };
  const project = ([x, y, z]) => [230 + x * 155 - z * 55, 290 - y * 145 - z * 45];

  function surface(u, v) {
    const a = lerp(control.p00, control.p10, u);
    const b = lerp(control.p01, control.p11, u);
    return lerp(a, b, v);
  }

  function derivatives(u, v) {
    return {
      du: add(scale(subtract(control.p10, control.p00), 1 - v), scale(subtract(control.p11, control.p01), v)),
      dv: add(scale(subtract(control.p01, control.p00), 1 - u), scale(subtract(control.p11, control.p10), u)),
    };
  }

  function drawMesh(resolution, color, lineWidth, fill) {
    const grid = Array.from({ length: resolution + 1 }, (_, row) =>
      Array.from({ length: resolution + 1 }, (_, column) => surface(column / resolution, row / resolution)));
    if (fill) {
      for (let row = 0; row < resolution; row += 1) {
        for (let column = 0; column < resolution; column += 1) {
          const corners = [grid[row][column], grid[row][column + 1], grid[row + 1][column + 1], grid[row + 1][column]];
          context.beginPath();
          corners.forEach((point, index) => {
            const screen = project(point);
            if (index === 0) context.moveTo(...screen);
            else context.lineTo(...screen);
          });
          context.closePath();
          context.fillStyle = fill;
          context.fill();
        }
      }
    }
    context.strokeStyle = color;
    context.lineWidth = lineWidth;
    for (let row = 0; row <= resolution; row += 1) {
      context.beginPath();
      grid[row].forEach((point, index) => {
        const screen = project(point);
        if (index === 0) context.moveTo(...screen);
        else context.lineTo(...screen);
      });
      context.stroke();
    }
    for (let column = 0; column <= resolution; column += 1) {
      context.beginPath();
      grid.forEach((row) => {
        const screen = row[column];
        if (column === 0 && row === grid[0]) context.moveTo(...screen);
        else context.lineTo(...screen);
      });
      context.stroke();
    }
  }

  function arrow(start, end, color) {
    const angle = Math.atan2(end[1] - start[1], end[0] - start[0]);
    context.strokeStyle = color;
    context.fillStyle = color;
    context.lineWidth = 3;
    context.beginPath();
    context.moveTo(...start);
    context.lineTo(...end);
    context.stroke();
    context.beginPath();
    context.moveTo(...end);
    context.lineTo(end[0] - 10 * Math.cos(angle - 0.45), end[1] - 10 * Math.sin(angle - 0.45));
    context.lineTo(end[0] - 10 * Math.cos(angle + 0.45), end[1] - 10 * Math.sin(angle + 0.45));
    context.closePath();
    context.fill();
    context.lineWidth = 1;
  }

  function update() {
    const u = Number(uInput.value);
    const v = Number(vInput.value);
    const mode = modes.find((input) => input.checked).value;
    const h = 0.001;
    const point = surface(u, v);
    const analytic = derivatives(u, v);
    const numeric = {
      du: scale(subtract(surface(u + h, v), surface(u - h, v)), 1 / (2 * h)),
      dv: scale(subtract(surface(u, v + h), surface(u, v - h)), 1 / (2 * h)),
    };
    const normal = normalize(cross(analytic.du, analytic.dv));
    const error = Math.max(Math.hypot(...subtract(analytic.du, numeric.du)), Math.hypot(...subtract(analytic.dv, numeric.dv)));
    uInput.nextElementSibling.value = u.toFixed(2);
    vInput.nextElementSibling.value = v.toFixed(2);

    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#10242c";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#9db6b8";
    context.font = "15px system-ui";
    context.fillText("2 × 2 双线性控制网格 · 一次边中点细分", 28, 30);

    if (mode !== "subdivided") drawMesh(1, "#789096", 2, "rgba(120,144,150,.10)");
    if (mode !== "raw") drawMesh(2, "#58c7bd", 2, "rgba(88,199,189,.08)");

    Object.entries(control).forEach(([label, position]) => {
      const screen = project(position);
      context.beginPath();
      context.arc(screen[0], screen[1], 5, 0, Math.PI * 2);
      context.fillStyle = "#ef9a67";
      context.fill();
      context.fillStyle = "#ef9a67";
      context.fillText(label, screen[0] + 7, screen[1] - 7);
    });

    const pointScreen = project(point);
    const normalEnd = project(add(point, scale(normal, 0.45)));
    arrow(pointScreen, normalEnd, "#ffd48a");
    context.beginPath();
    context.arc(pointScreen[0], pointScreen[1], 6, 0, Math.PI * 2);
    context.fillStyle = "#ffd48a";
    context.fill();
    context.fillStyle = "#ffd48a";
    context.fillText("S(u,v)", pointScreen[0] + 9, pointScreen[1] - 8);
    context.fillText("n", normalEnd[0] + 7, normalEnd[1] - 5);

    const fmt = (vector) => `(${vector.map((value) => window.RTBook.formatNumber(value, 3)).join(", ")})`;
    const counts = mode === "raw" ? "V=4 · F=1" : mode === "subdivided" ? "V=9 · F=4" : "原始 V=4/F=1，细分 V=9/F=4";
    readout.textContent =
      `u=${u.toFixed(2)} · v=${v.toFixed(2)} · S=${fmt(point)} · Su=${fmt(analytic.du)} · Sv=${fmt(analytic.dv)} · ` +
      `n=${fmt(normal)} · 有限差分最大误差=${window.RTBook.formatNumber(error, 6)} · ${counts}`;
  }

  uInput.addEventListener("input", update);
  vInput.addEventListener("input", update);
  modes.forEach((input) => input.addEventListener("change", update));
  reset.addEventListener("click", () => {
    uInput.value = "0.5";
    vInput.value = "0.5";
    modes[0].checked = true;
    update();
  });
  update();
})();
