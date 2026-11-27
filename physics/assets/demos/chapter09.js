(() => {
  const mount = document.getElementById("scene-demo");
  if (!mount) return;

  mount.insertAdjacentHTML(
    "beforeend",
    `<div class="demo-toolbar">
      <label>父平移 X <input type="range" data-parent-x min="-2" max="4" step="0.1" value="2"><output>2.0</output></label>
      <label>子旋转 <input type="range" data-angle min="-180" max="180" step="1" value="90"><output>90°</output></label>
      <label><input type="radio" name="bound-mode" value="aabb" checked> AABB</label>
      <label><input type="radio" name="bound-mode" value="sphere"> 包围球</label>
      <button type="button" data-reset>重置</button>
    </div>
    <canvas id="scene-canvas" width="720" height="360" role="img" aria-label="父子节点变换、局部点到世界点和世界空间包围体"></canvas>
    <p class="demo-readout" data-readout aria-live="polite"></p>`
  );

  const canvas = mount.querySelector("#scene-canvas");
  const context = canvas.getContext("2d");
  const readout = mount.querySelector("[data-readout]");
  const parentInput = mount.querySelector("[data-parent-x]");
  const angleInput = mount.querySelector("[data-angle]");
  const modes = [...mount.querySelectorAll('[name="bound-mode"]')];
  const reset = mount.querySelector("[data-reset]");
  const localCorners = [[0, 0], [1, 0], [1, 1], [0, 1]];
  const localPoint = [1, 0];
  const frustum = { minX: -0.5, maxX: 2.5, minY: -1, maxY: 2 };
  const view = { centerX: 355, centerY: 205, scale: 92 };
  const toScreen = ([x, y]) => [view.centerX + x * view.scale, view.centerY - y * view.scale];
  const fmt = (value) => window.RTBook.formatNumber(value, 3);
  const fmtPoint = (point) => `(${point.map(fmt).join(", ")})`;

  function polygon(points, stroke, fill, width = 2, dash = []) {
    context.beginPath();
    points.forEach((point, index) => {
      const screen = toScreen(point);
      if (index === 0) context.moveTo(...screen);
      else context.lineTo(...screen);
    });
    context.closePath();
    if (fill) { context.fillStyle = fill; context.fill(); }
    context.strokeStyle = stroke;
    context.lineWidth = width;
    context.setLineDash(dash);
    context.stroke();
    context.setLineDash([]);
    context.lineWidth = 1;
  }

  function marker(point, color, radius = 6) {
    const screen = toScreen(point);
    context.beginPath();
    context.arc(screen[0], screen[1], radius, 0, Math.PI * 2);
    context.fillStyle = color;
    context.fill();
  }

  function update() {
    const tx = Number(parentInput.value);
    const degrees = Number(angleInput.value);
    const angle = degrees * Math.PI / 180;
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);
    const rotate = ([x, y]) => [x * cosine - y * sine, x * sine + y * cosine];
    const translate = ([x, y]) => [x + tx, y];
    const worldPoint = translate(rotate(localPoint));
    const worldCorners = localCorners.map((point) => translate(rotate(point)));
    const minimum = [Math.min(...worldCorners.map((point) => point[0])), Math.min(...worldCorners.map((point) => point[1]))];
    const maximum = [Math.max(...worldCorners.map((point) => point[0])), Math.max(...worldCorners.map((point) => point[1]))];
    const sphereCenter = translate(rotate([0.5, 0.5]));
    const sphereRadius = Math.SQRT1_2;
    const mode = modes.find((input) => input.checked).value;
    const aabbVisible = minimum[0] <= frustum.maxX && maximum[0] >= frustum.minX && minimum[1] <= frustum.maxY && maximum[1] >= frustum.minY;
    const closest = [window.RTBook.clamp(sphereCenter[0], frustum.minX, frustum.maxX), window.RTBook.clamp(sphereCenter[1], frustum.minY, frustum.maxY)];
    const sphereVisible = Math.hypot(sphereCenter[0] - closest[0], sphereCenter[1] - closest[1]) <= sphereRadius;
    const visible = mode === "aabb" ? aabbVisible : sphereVisible;
    parentInput.nextElementSibling.value = tx.toFixed(1);
    angleInput.nextElementSibling.value = `${degrees}°`;

    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#10242c";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#9db6b8";
    context.font = "15px system-ui";
    context.fillText("虚线矩形是二维视锥粗检区域；包围体只用于排除，不代表最终可见", 28, 30);

    context.strokeStyle = "rgba(255,255,255,.12)";
    context.beginPath();
    context.moveTo(40, view.centerY);
    context.lineTo(680, view.centerY);
    context.moveTo(view.centerX, 55);
    context.lineTo(view.centerX, 330);
    context.stroke();
    polygon([[frustum.minX, frustum.minY], [frustum.maxX, frustum.minY], [frustum.maxX, frustum.maxY], [frustum.minX, frustum.maxY]], "#8fb2b6", null, 2, [7, 6]);
    polygon(worldCorners, "#58c7bd", "rgba(88,199,189,.18)", 3);

    const parentOrigin = toScreen([tx, 0]);
    const axisX = toScreen([tx + cosine * 0.65, sine * 0.65]);
    const axisY = toScreen([tx - sine * 0.65, cosine * 0.65]);
    context.strokeStyle = "#ef9a67";
    context.lineWidth = 2;
    context.beginPath(); context.moveTo(...parentOrigin); context.lineTo(...axisX); context.stroke();
    context.strokeStyle = "#ffd48a";
    context.beginPath(); context.moveTo(...parentOrigin); context.lineTo(...axisY); context.stroke();
    context.lineWidth = 1;
    marker([tx, 0], "#ef9a67", 7);
    context.fillStyle = "#ef9a67";
    context.fillText("父节点原点", parentOrigin[0] + 10, parentOrigin[1] + 18);

    if (mode === "aabb") {
      polygon([[minimum[0], minimum[1]], [maximum[0], minimum[1]], [maximum[0], maximum[1]], [minimum[0], maximum[1]]], "#ffd48a", null, 2, [5, 4]);
    } else {
      const center = toScreen(sphereCenter);
      context.beginPath();
      context.arc(center[0], center[1], sphereRadius * view.scale, 0, Math.PI * 2);
      context.strokeStyle = "#ffd48a";
      context.lineWidth = 2;
      context.setLineDash([5, 4]);
      context.stroke();
      context.setLineDash([]);
      context.lineWidth = 1;
    }

    const childPoint = rotate(localPoint);
    marker(localPoint, "#9db6b8", 5);
    marker(worldPoint, "#ffd48a", 7);
    const localScreen = toScreen(localPoint);
    const worldScreen = toScreen(worldPoint);
    context.fillStyle = "#9db6b8";
    context.fillText("p_local", localScreen[0] - 60, localScreen[1] + 22);
    context.fillStyle = "#ffd48a";
    context.fillText("p_world", worldScreen[0] + 10, worldScreen[1] - 8);

    const matrix = `[[${fmt(cosine)}, ${fmt(-sine)}, ${fmt(tx)}], [${fmt(sine)}, ${fmt(cosine)}, 0], [0, 0, 1]]`;
    const bounds = mode === "aabb"
      ? `AABB=[${fmt(minimum[0])}, ${fmt(maximum[0])}] × [${fmt(minimum[1])}, ${fmt(maximum[1])}]`
      : `球心=${fmtPoint(sphereCenter)} · r=${fmt(sphereRadius)}`;
    readout.textContent =
      `父平移=(${fmt(tx)}, 0) · 子旋转=${degrees}° · p_local=${fmtPoint(localPoint)} · ` +
      `子节点坐标=${fmtPoint(childPoint)} · p_world=${fmtPoint(worldPoint)} · ` +
      `${bounds} · 视锥粗检=${visible ? "可能相交，继续处理" : "明确在外，可剔除"} · M_world=${matrix}`;
  }

  parentInput.addEventListener("input", update);
  angleInput.addEventListener("input", update);
  modes.forEach((input) => input.addEventListener("change", update));
  reset.addEventListener("click", () => {
    parentInput.value = "2";
    angleInput.value = "90";
    modes[0].checked = true;
    update();
  });
  update();
})();
