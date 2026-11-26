(() => {
  const mount = document.getElementById("transform-demo");
  if (!mount) return;

  const controls = [
    ["sx", "缩放 X", 0.25, 2.5, 1.6, 0.05],
    ["sy", "缩放 Y", 0.25, 2.5, 0.8, 0.05],
    ["angle", "绕 Z 旋转", -180, 180, 35, 1],
    ["tx", "平移 X", -4, 4, 1.2, 0.1],
    ["ty", "平移 Y", -4, 4, -0.6, 0.1],
  ];
  mount.insertAdjacentHTML(
    "beforeend",
    `<div class="demo-toolbar">${controls.map(([name, label, min, max, value, step]) =>
      `<label>${label}<input type="range" data-control="${name}" min="${min}" max="${max}" step="${step}" value="${value}"><output>${value}</output></label>`
    ).join("")}<button type="button" data-reset>重置</button></div>
    <canvas id="transform-canvas" width="640" height="380" role="img" aria-label="局部正方形经过缩放、旋转和平移后的位置"></canvas>
    <p class="demo-readout" data-readout aria-live="polite"></p>`
  );

  const canvas = mount.querySelector("#transform-canvas");
  const context = canvas.getContext("2d");
  const inputs = [...mount.querySelectorAll("[data-control]")];
  const readout = mount.querySelector("[data-readout]");
  const origin = { x: 250, y: 240 };
  const scale = 58;
  const point = [1, 0.5, 0];
  const normal = [1, 0, 0];

  function values() {
    return Object.fromEntries(inputs.map((input) => [input.dataset.control, Number(input.value)]));
  }

  function rotate(x, y, angle) {
    const radians = (angle * Math.PI) / 180;
    return [x * Math.cos(radians) - y * Math.sin(radians), x * Math.sin(radians) + y * Math.cos(radians)];
  }

  function transform(x, y, parameters) {
    const scaled = [x * parameters.sx, y * parameters.sy];
    const rotated = rotate(scaled[0], scaled[1], parameters.angle);
    return [rotated[0] + parameters.tx, rotated[1] + parameters.ty];
  }

  function pathPolygon(vertices, color, dashed = false) {
    context.save();
    context.strokeStyle = color;
    context.lineWidth = dashed ? 2 : 3;
    if (dashed) context.setLineDash([6, 5]);
    context.beginPath();
    vertices.forEach(([x, y], index) => {
      const screenX = origin.x + x * scale;
      const screenY = origin.y - y * scale;
      if (index === 0) context.moveTo(screenX, screenY);
      else context.lineTo(screenX, screenY);
    });
    context.closePath();
    context.stroke();
    context.restore();
  }

  function arrowVector(vector, color, label) {
    const [x, y] = vector;
    context.save();
    context.strokeStyle = color;
    context.fillStyle = color;
    context.lineWidth = 3;
    context.beginPath();
    context.moveTo(origin.x, origin.y);
    context.lineTo(origin.x + x * scale, origin.y - y * scale);
    context.stroke();
    context.beginPath();
    context.arc(origin.x + x * scale, origin.y - y * scale, 5, 0, Math.PI * 2);
    context.fill();
    context.font = "600 15px system-ui";
    context.fillText(label, origin.x + x * scale + 8, origin.y - y * scale - 8);
    context.restore();
  }

  function update() {
    const parameters = values();
    inputs.forEach((input) => { input.nextElementSibling.value = input.value; });
    const localVertices = [[-0.6, -0.6], [0.6, -0.6], [0.6, 0.6], [-0.6, 0.6]];
    const transformedVertices = localVertices.map(([x, y]) => transform(x, y, parameters));
    const transformedPoint = transform(point[0], point[1], parameters);
    const scaledNormal = [normal[0] / parameters.sx, normal[1] / parameters.sy];
    const transformedNormal = rotate(scaledNormal[0], scaledNormal[1], parameters.angle);

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
    context.fillText("虚线：局部正方形；实线：T Rz S 变换后", 24, 26);
    context.fillText("x", canvas.width - 24, origin.y - 10);
    context.fillText("y", origin.x + 10, 24);
    pathPolygon(localVertices, "#607e84", true);
    pathPolygon(transformedVertices, "#58c7bd");
    arrowVector(transformedPoint, "#ef9a67", "p'");
    arrowVector(transformedNormal, "#ffd48a", "n'");

    const fmt = (value) => window.RTBook.formatNumber(value, 3);
    readout.textContent =
      `M = T Rz(${parameters.angle}°) S(${fmt(parameters.sx)}, ${fmt(parameters.sy)}) · ` +
      `p'=(${transformedPoint.map(fmt).join(", ")}, 0) · ` +
      `n'=(${transformedNormal.map(fmt).join(", ")}, 0)（归一化前）`;
  }

  inputs.forEach((input) => input.addEventListener("input", update));
  mount.querySelector("[data-reset]").addEventListener("click", () => {
    inputs.forEach((input, index) => { input.value = String(controls[index][4]); });
    update();
  });
  update();
})();
