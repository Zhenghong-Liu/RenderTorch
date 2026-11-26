(() => {
  const mount = document.getElementById("camera-demo");
  if (!mount) return;

  const controls = [
    ["eye-x", "眼睛 X", -8, 8, 3, 0.1],
    ["eye-y", "眼睛 Y", -4, 8, 2, 0.1],
    ["eye-z", "眼睛 Z", 1, 10, 6, 0.1],
    ["point-x", "点 X", -5, 5, 1.5, 0.1],
    ["point-y", "点 Y", -4, 6, 1, 0.1],
    ["point-z", "点 Z", -4, 4, 3, 0.1],
    ["fov", "垂直 fov", 30, 100, 60, 1],
  ];
  mount.insertAdjacentHTML(
    "beforeend",
    `<div class="demo-toolbar">${controls.map(([name, label, min, max, value, step]) =>
      `<label>${label}<input type="range" data-camera="${name}" min="${min}" max="${max}" step="${step}" value="${value}"><output>${value}</output></label>`
    ).join("")}<button type="button" data-reset>重置</button></div>
    <canvas id="camera-canvas" width="720" height="320" role="img" aria-label="相机与世界点的俯视示意，以及 NDC 到屏幕像素的映射"></canvas>
    <p class="demo-readout" data-readout aria-live="polite"></p>`
  );

  const canvas = mount.querySelector("#camera-canvas");
  const context = canvas.getContext("2d");
  const inputs = [...mount.querySelectorAll("[data-camera]")];
  const readout = mount.querySelector("[data-readout]");
  const width = 512;
  const height = 256;

  const subtract = (a, b) => a.map((value, index) => value - b[index]);
  const dot = (a, b) => a.reduce((sum, value, index) => sum + value * b[index], 0);
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const normalize = (vector) => {
    const magnitude = Math.sqrt(dot(vector, vector));
    return magnitude > 1e-12 ? vector.map((value) => value / magnitude) : null;
  };
  const multiply = (matrix, vector) => matrix.map((row) => row.reduce((sum, value, index) => sum + value * vector[index], 0));

  function marker(x, y, color, radius = 5) {
    context.fillStyle = color;
    context.beginPath();
    context.arc(x, y, radius, 0, Math.PI * 2);
    context.fill();
  }

  function update() {
    const parameters = Object.fromEntries(inputs.map((input) => [input.dataset.camera, Number(input.value)]));
    inputs.forEach((input) => { input.nextElementSibling.value = input.value; });
    const eye = [parameters["eye-x"], parameters["eye-y"], parameters["eye-z"]];
    const target = [0, 0, 0];
    const point = [parameters["point-x"], parameters["point-y"], parameters["point-z"]];
    const forward = normalize(subtract(target, eye));
    const right = forward && normalize(cross(forward, [0, 1, 0]));
    const up = right && cross(right, forward);

    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#10242c";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#9db6b8";
    context.font = "14px system-ui";
    context.fillText("世界俯视（x-z）", 24, 26);
    context.fillText("屏幕像素（512 × 256）", 420, 26);

    const worldOrigin = { x: 155, y: 165 };
    const worldScale = 18;
    const toWorld = ([x, , z]) => [worldOrigin.x + x * worldScale, worldOrigin.y + z * worldScale];
    context.strokeStyle = "rgba(255,255,255,.16)";
    context.beginPath(); context.moveTo(24, worldOrigin.y); context.lineTo(300, worldOrigin.y); context.stroke();
    context.beginPath(); context.moveTo(worldOrigin.x, 60); context.lineTo(worldOrigin.x, 280); context.stroke();
    const eyeScreen = toWorld(eye);
    const pointScreen = toWorld(point);
    context.strokeStyle = "#58c7bd";
    context.lineWidth = 2;
    context.beginPath(); context.moveTo(...eyeScreen); context.lineTo(...toWorld(target)); context.stroke();
    marker(...eyeScreen, "#ef9a67", 7);
    marker(...pointScreen, "#ffd48a", 6);
    context.fillStyle = "#ef9a67"; context.fillText("E", eyeScreen[0] + 9, eyeScreen[1] - 8);
    context.fillStyle = "#ffd48a"; context.fillText("p", pointScreen[0] + 9, pointScreen[1] - 8);
    marker(...toWorld(target), "#9db6b8", 4);
    context.fillStyle = "#9db6b8"; context.fillText("C", worldOrigin.x + 8, worldOrigin.y + 20);

    const viewport = { x: 370, y: 72, width: 300, height: 150 };
    context.strokeStyle = "#77949a";
    context.strokeRect(viewport.x, viewport.y, viewport.width, viewport.height);
    context.beginPath();
    context.moveTo(viewport.x + viewport.width / 2, viewport.y);
    context.lineTo(viewport.x + viewport.width / 2, viewport.y + viewport.height);
    context.moveTo(viewport.x, viewport.y + viewport.height / 2);
    context.lineTo(viewport.x + viewport.width, viewport.y + viewport.height / 2);
    context.stroke();

    const fmt = (value) => window.RTBook.formatNumber(value, 3);
    if (!right || !up) {
      readout.textContent = "相机视线与世界上方向共线，lookAt 退化；请调整眼睛位置。";
      return;
    }
    const view = [
      [...right, -dot(right, eye)],
      [...up, -dot(up, eye)],
      [...forward.map((value) => -value), dot(forward, eye)],
      [0, 0, 0, 1],
    ];
    const viewPoint = multiply(view, [...point, 1]).slice(0, 3);
    const fov = (parameters.fov * Math.PI) / 180;
    const focal = 1 / Math.tan(fov / 2);
    const aspect = width / height;
    const wClip = -viewPoint[2];
    const near = 0.1;
    const far = 100;
    const projection = [
      [focal / aspect, 0, 0, 0],
      [0, focal, 0, 0],
      [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
      [0, 0, -1, 0],
    ];
    const clip = multiply(projection, [...viewPoint, 1]);
    if (wClip <= 0) {
      readout.textContent = `view=(${viewPoint.map(fmt).join(", ")}) · w_clip=${fmt(wClip)} ≤ 0：相机平面上或背后，必须裁剪。`;
      return;
    }
    const ndc = clip.slice(0, 3).map((value) => value / clip[3]);
    const pixel = [(ndc[0] + 1) * width / 2, (1 - ndc[1]) * height / 2, ndc[2]];
    const inside = Math.abs(ndc[0]) <= 1 && Math.abs(ndc[1]) <= 1 && ndc[2] >= -1 && ndc[2] <= 1;
    const markerX = viewport.x + ((pixel[0] / width) * viewport.width);
    const markerY = viewport.y + ((pixel[1] / height) * viewport.height);
    context.save();
    context.beginPath();
    context.rect(viewport.x, viewport.y, viewport.width, viewport.height);
    context.clip();
    marker(markerX, markerY, inside ? "#58c7bd" : "#ef9a67", 7);
    context.restore();
    context.fillStyle = inside ? "#58c7bd" : "#ef9a67";
    context.fillText(inside ? "位于裁剪范围内" : "位于裁剪范围外", 420, 248);
    readout.textContent =
      `view=(${viewPoint.map(fmt).join(", ")}) · ` +
      `NDC=(${ndc.map(fmt).join(", ")}) · ` +
      `pixel=(${pixel.slice(0, 2).map(fmt).join(", ")}) · z_ndc=${fmt(ndc[2])}`;
  }

  inputs.forEach((input) => input.addEventListener("input", update));
  mount.querySelector("[data-reset]").addEventListener("click", () => {
    inputs.forEach((input, index) => { input.value = String(controls[index][4]); });
    update();
  });
  update();
})();
