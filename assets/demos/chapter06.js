(() => {
  const mount = document.getElementById("topology-demo");
  if (!mount) return;

  mount.insertAdjacentHTML(
    "beforeend",
    `<div class="demo-toolbar">
      <label><input type="radio" name="normal-mode" value="face" checked> 面法线</label>
      <label><input type="radio" name="normal-mode" value="vertex"> 顶点法线</label>
      <label><input type="checkbox" data-flip> 翻转面 0 绕序</label>
      <button type="button" data-reset>重置</button>
    </div>
    <canvas id="topology-canvas" width="720" height="360" role="img" aria-label="四面体拓扑与面法线或顶点法线"></canvas>
    <p class="demo-readout" data-readout aria-live="polite"></p>`
  );

  const canvas = mount.querySelector("#topology-canvas");
  const context = canvas.getContext("2d");
  const readout = mount.querySelector("[data-readout]");
  const normalModes = [...mount.querySelectorAll('[name="normal-mode"]')];
  const flip = mount.querySelector("[data-flip]");
  const reset = mount.querySelector("[data-reset]");
  const vertices = [
    [0, 0, 0],
    [1.4, 0, 0],
    [0.65, 1.15, 0],
    [0.45, 0.45, 1.25],
  ];
  const baseFaces = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]];
  const labels = ["A", "B", "C", "D"];

  const subtract = (a, b) => a.map((value, index) => value - b[index]);
  const add = (a, b) => a.map((value, index) => value + b[index]);
  const scale = (a, factor) => a.map((value) => value * factor);
  const dot = (a, b) => a.reduce((sum, value, index) => sum + value * b[index], 0);
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const normalize = (vector) => {
    const length = Math.hypot(...vector);
    return length > 1e-12 ? scale(vector, 1 / length) : [0, 0, 0];
  };
  const project = ([x, y, z]) => [235 + x * 185 + z * 70, 292 - y * 175 - z * 55];

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
    const faces = baseFaces.map((face) => [...face]);
    if (flip.checked) faces[0] = [faces[0][0], faces[0][2], faces[0][1]];
    const edges = new Set();
    faces.forEach((face) => {
      for (let index = 0; index < 3; index += 1) {
        edges.add([face[index], face[(index + 1) % 3]].sort((a, b) => a - b).join("-"));
      }
    });
    const rawNormals = faces.map((face) => cross(subtract(vertices[face[1]], vertices[face[0]]), subtract(vertices[face[2]], vertices[face[0]])));
    const faceNormals = rawNormals.map(normalize);
    const vertexNormals = vertices.map(() => [0, 0, 0]);
    faces.forEach((face, faceIndex) => {
      face.forEach((vertexIndex) => {
        vertexNormals[vertexIndex] = add(vertexNormals[vertexIndex], rawNormals[faceIndex]);
      });
    });
    vertexNormals.forEach((normal, index) => { vertexNormals[index] = normalize(normal); });
    const faceMode = normalModes.find((input) => input.checked).value === "face";

    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#10242c";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#9db6b8";
    context.font = "15px system-ui";
    context.fillText("三维叉积在投影前计算；屏幕箭头只是观察方向", 28, 30);

    context.beginPath();
    faces[0].forEach((vertexIndex, index) => {
      const point = project(vertices[vertexIndex]);
      if (index === 0) context.moveTo(...point);
      else context.lineTo(...point);
    });
    context.closePath();
    context.fillStyle = "rgba(88,199,189,.22)";
    context.fill();

    edges.forEach((edge) => {
      const [a, b] = edge.split("-").map(Number);
      context.beginPath();
      context.moveTo(...project(vertices[a]));
      context.lineTo(...project(vertices[b]));
      context.strokeStyle = "#8fb2b6";
      context.lineWidth = 2;
      context.stroke();
    });

    const normalData = faceMode
      ? faces.map((face, index) => {
          const center = face.reduce((sum, vertexIndex) => add(sum, vertices[vertexIndex]), [0, 0, 0]).map((value) => value / 3);
          return { origin: center, normal: faceNormals[index], label: `n${index}` };
        })
      : vertexNormals.map((normal, index) => ({ origin: vertices[index], normal, label: `n${labels[index]}` }));
    normalData.forEach(({ origin, normal, label }) => {
      const start = project(origin);
      const end = project(add(origin, scale(normal, 0.55)));
      arrow(start, end, "#ef9a67");
      context.fillStyle = "#ef9a67";
      context.fillText(label, end[0] + 6, end[1] - 5);
    });

    vertices.forEach((vertex, index) => {
      const point = project(vertex);
      context.beginPath();
      context.arc(point[0], point[1], 7, 0, Math.PI * 2);
      context.fillStyle = "#58c7bd";
      context.fill();
      context.fillStyle = "#edf5f3";
      context.fillText(labels[index], point[0] + 10, point[1] - 8);
    });

    const fmt = (vector) => `(${vector.map((value) => window.RTBook.formatNumber(value, 3)).join(", ")})`;
    readout.textContent =
      `V=${vertices.length} · E=${edges.size} · F=${faces.length} · V-E+F=${vertices.length - edges.size + faces.length} · ` +
      `面0绕序=${flip.checked ? "已翻转" : "原始"} · ${faceMode ? "面法线" : "顶点法线"} ` +
      `${faceMode ? "n0" : "nA"}=${fmt(faceMode ? faceNormals[0] : vertexNormals[0])}`;
  }

  normalModes.forEach((input) => input.addEventListener("change", update));
  flip.addEventListener("change", update);
  reset.addEventListener("click", () => {
    normalModes[0].checked = true;
    flip.checked = false;
    update();
  });
  update();
})();
