(() => {
  const mount = document.getElementById("vector-demo");
  if (!mount) return;

  const names = ["ax", "ay", "az", "bx", "by", "bz"];
  const initial = [2, 2, 1, 3, -1, 2];
  mount.insertAdjacentHTML(
    "beforeend",
    `<div class="demo-toolbar">${names.map((name, index) =>
      `<label>${name.toUpperCase()} <input type="range" data-vector="${index}" min="-5" max="5" step="0.1" value="${initial[index]}"><output>${initial[index]}</output></label>`
    ).join("")}<button type="button" data-reset>重置</button></div>
    <canvas id="vector-canvas" width="640" height="340" role="img" aria-label="两个向量及其点积投影的二维示意图"></canvas>
    <p class="demo-readout" data-readout aria-live="polite"></p>`
  );

  const canvas = mount.querySelector("#vector-canvas");
  const context = canvas.getContext("2d");
  const inputs = [...mount.querySelectorAll("[data-vector]")];
  const readout = mount.querySelector("[data-readout]");
  const origin = { x: 135, y: 245 };
  const scale = 30;

  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const length = (a) => Math.sqrt(dot(a, a));
  const cross = (a, b) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];

  function arrow(x, y, color, label, dashed = false) {
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    context.save();
    context.strokeStyle = color;
    context.fillStyle = color;
    context.lineWidth = 3;
    if (dashed) context.setLineDash([7, 5]);
    context.beginPath();
    context.moveTo(origin.x, origin.y);
    context.lineTo(origin.x + x * scale, origin.y - y * scale);
    context.stroke();
    context.setLineDash([]);
    const angle = Math.atan2(-y, x);
    const tipX = origin.x + x * scale;
    const tipY = origin.y - y * scale;
    context.translate(tipX, tipY);
    context.rotate(angle);
    context.beginPath();
    context.moveTo(0, 0);
    context.lineTo(-10, -5);
    context.lineTo(-10, 5);
    context.closePath();
    context.fill();
    context.restore();
    context.fillStyle = color;
    context.font = "600 16px system-ui";
    context.fillText(label, origin.x + x * scale + 8, origin.y - y * scale - 8);
  }

  function update() {
    const a = inputs.slice(0, 3).map((input) => Number(input.value));
    const b = inputs.slice(3).map((input) => Number(input.value));
    inputs.forEach((input) => { input.nextElementSibling.value = input.value; });
    const aLength = length(a);
    const bLength = length(b);
    const scalar = dot(a, b);
    const product = cross(a, b);
    const cosine = aLength && bLength ? Math.max(-1, Math.min(1, scalar / (aLength * bLength))) : null;
    const angle = cosine === null ? null : (Math.acos(cosine) * 180) / Math.PI;
    const aSquared = dot(a, a);
    const factor = aSquared > 1e-12 ? scalar / aSquared : 0;
    const projection = a.map((value) => value * factor);

    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#10242c";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.strokeStyle = "rgba(255,255,255,.12)";
    context.lineWidth = 1;
    for (let x = origin.x; x < canvas.width; x += scale) {
      context.beginPath(); context.moveTo(x, 0); context.lineTo(x, canvas.height); context.stroke();
    }
    for (let y = origin.y; y >= 0; y -= scale) {
      context.beginPath(); context.moveTo(0, y); context.lineTo(canvas.width, y); context.stroke();
    }
    context.strokeStyle = "#77949a";
    context.beginPath(); context.moveTo(0, origin.y); context.lineTo(canvas.width, origin.y); context.stroke();
    context.beginPath(); context.moveTo(origin.x, canvas.height); context.lineTo(origin.x, 0); context.stroke();
    context.fillStyle = "#9db6b8";
    context.font = "14px system-ui";
    context.fillText("图中仅显示 x-y 分量；z 分量参与全部三维读数", 28, 26);
    context.fillText("x", canvas.width - 24, origin.y - 10);
    context.fillText("y", origin.x + 10, 24);
    if (aSquared > 1e-12) arrow(projection[0], projection[1], "#b7d8d4", "proj", true);
    arrow(a[0], a[1], "#58c7bd", "a");
    arrow(b[0], b[1], "#ef9a67", "b");

    const fmt = (value) => window.RTBook.formatNumber(value, 3);
    readout.textContent =
      `a·b=${fmt(scalar)} · ‖a‖=${fmt(aLength)} · ‖b‖=${fmt(bLength)} · ` +
      `θ=${angle === null ? "未定义" : `${fmt(angle)}°`} · ` +
      `a×b=(${product.map(fmt).join(", ")}) · ‖a×b‖=${fmt(length(product))}`;
  }

  inputs.forEach((input) => input.addEventListener("input", update));
  mount.querySelector("[data-reset]").addEventListener("click", () => {
    inputs.forEach((input, index) => { input.value = String(initial[index]); });
    update();
  });
  update();
})();
