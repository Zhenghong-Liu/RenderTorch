(() => {
  const mount = document.getElementById("mesh-demo");
  if (!mount) return;

  mount.insertAdjacentHTML(
    "beforeend",
    `<div class="demo-toolbar">
      <label><input type="radio" name="mesh-mode" value="fan" checked> 三角形扇</label>
      <label><input type="radio" name="mesh-mode" value="strip"> 三角形带</label>
      <label><input type="radio" name="mesh-mode" value="indexed"> 显式索引</label>
      <button type="button" data-reset>重置</button>
    </div>
    <canvas id="mesh-canvas" width="720" height="330" role="img" aria-label="同一四边形的三角形扇、三角形带和显式索引连接"></canvas>
    <p class="demo-readout" data-readout aria-live="polite"></p>`
  );

  const canvas = mount.querySelector("#mesh-canvas");
  const context = canvas.getContext("2d");
  const readout = mount.querySelector("[data-readout]");
  const modes = [...mount.querySelectorAll('[name="mesh-mode"]')];
  const reset = mount.querySelector("[data-reset]");
  const points = [
    { x: 180, y: 250 },
    { x: 520, y: 250 },
    { x: 520, y: 80 },
    { x: 180, y: 80 },
  ];

  function trianglePath(face) {
    context.beginPath();
    face.forEach((index, position) => {
      const point = points[index];
      if (position === 0) context.moveTo(point.x, point.y);
      else context.lineTo(point.x, point.y);
    });
    context.closePath();
  }

  function update() {
    const mode = modes.find((input) => input.checked).value;
    const stripOrder = [0, 1, 3, 2];
    const faces = mode === "strip" ? [[0, 1, 2], [2, 1, 3]] : [[0, 1, 2], [0, 2, 3]];
    const edges = new Map();
    faces.forEach((face) => {
      for (let index = 0; index < 3; index += 1) {
        const a = face[index];
        const b = face[(index + 1) % 3];
        const key = [a, b].sort((left, right) => left - right).join("-");
        edges.set(key, (edges.get(key) || 0) + 1);
      }
    });

    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#10242c";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#9db6b8";
    context.font = "15px system-ui";
    context.fillText(mode === "strip" ? "条带顶点顺序：p0 → p1 → p3 → p2" : "原始顶点顺序：p0 → p1 → p2 → p3", 28, 30);

    faces.forEach((face, index) => {
      trianglePath(face);
      context.fillStyle = index === 0 ? "rgba(88,199,189,.18)" : "rgba(239,154,103,.18)";
      context.fill();
    });

    edges.forEach((count, key) => {
      const [a, b] = key.split("-").map(Number);
      context.beginPath();
      context.moveTo(points[a].x, points[a].y);
      context.lineTo(points[b].x, points[b].y);
      context.strokeStyle = count > 1 ? "#ef9a67" : "#8fb2b6";
      context.lineWidth = count > 1 ? 4 : 2;
      context.stroke();
    });

    points.forEach((point, index) => {
      context.beginPath();
      context.arc(point.x, point.y, mode === "fan" && index === 0 ? 9 : 7, 0, Math.PI * 2);
      context.fillStyle = mode === "fan" && index === 0 ? "#ffd48a" : "#58c7bd";
      context.fill();
      context.fillStyle = "#edf5f3";
      context.font = "15px system-ui";
      const label = mode === "strip" ? `v${stripOrder.indexOf(index)}` : `p${index}`;
      context.fillText(label, point.x + 12, point.y - 10);
    });

    context.fillStyle = "#ef9a67";
    context.fillText("橙色边被两个三角形共享", 28, 306);
    const sharedEdges = [...edges.values()].filter((count) => count > 1).length;
    const modeName = { fan: "三角形扇", strip: "三角形带", indexed: "显式索引" }[mode];
    readout.textContent =
      `${modeName} · 顶点记录=4 · 三角形=2 · 顶点引用=6 · ` +
      `边=${edges.size} · 共享内部边=${sharedEdges} · ` +
      `索引项=${mode === "indexed" ? "6 个显式整数" : "隐式条带/扇形指令"}`;
  }

  modes.forEach((input) => input.addEventListener("change", update));
  reset.addEventListener("click", () => {
    modes[0].checked = true;
    update();
  });
  update();
})();
