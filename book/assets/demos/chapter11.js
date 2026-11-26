(() => {
  const mount = document.getElementById("occlusion-demo");
  if (!mount) return;

  mount.insertAdjacentHTML(
    "beforeend",
    `<div class="demo-toolbar">
      <label>屏幕参数 t <input type="range" data-interp="t" min="0" max="1" step="0.01" value="0.5"><output>0.50</output></label>
      <label>近端 w <input type="range" data-interp="near" min="1" max="6" step="0.1" value="1"><output>1.0</output></label>
      <label>远端 w <input type="range" data-interp="far" min="2" max="10" step="0.1" value="4"><output>4.0</output></label>
      <label>绘制顺序<select data-order><option value="AB">A → B</option><option value="BA">B → A</option></select></label>
      <label>A 深度 <input type="range" data-depth="a" min="1" max="8" step="0.1" value="3"><output>3.0</output></label>
      <label>B 深度 <input type="range" data-depth="b" min="1" max="8" step="0.1" value="5"><output>5.0</output></label>
      <button type="button" data-reset>重置</button>
    </div>
    <div class="demo-grid">
      <figure><canvas id="linear-canvas" width="320" height="260" role="img" aria-label="直接线性插值 w 与属性 q"></canvas><figcaption>直接插值：屏幕线性，但不是透视正确</figcaption></figure>
      <figure><canvas id="inverse-canvas" width="320" height="260" role="img" aria-label="插值逆深度后恢复 w 与属性 q"></canvas><figcaption>插值 1/w：先插值倒数，再相除</figcaption></figure>
    </div>
    <canvas id="order-canvas" width="640" height="280" role="img" aria-label="画家算法与 z-buffer 在 A B 两种顺序下的结果"></canvas>
    <p class="demo-readout" data-readout aria-live="polite"></p>`
  );

  const linearCanvas = mount.querySelector("#linear-canvas");
  const inverseCanvas = mount.querySelector("#inverse-canvas");
  const orderCanvas = mount.querySelector("#order-canvas");
  const linearContext = linearCanvas.getContext("2d");
  const inverseContext = inverseCanvas.getContext("2d");
  const orderContext = orderCanvas.getContext("2d");
  const interpolationInputs = [...mount.querySelectorAll("[data-interp]")];
  const depthInputs = [...mount.querySelectorAll("[data-depth]")];
  const orderSelect = mount.querySelector("[data-order]");
  const readout = mount.querySelector("[data-readout]");
  const fmt = (value) => window.RTBook.formatNumber(value, 3);

  function values() {
    const parameters = Object.fromEntries(interpolationInputs.map((input) => [input.dataset.interp, Number(input.value)]));
    const depths = Object.fromEntries(depthInputs.map((input) => [input.dataset.depth, Number(input.value)]));
    const t = parameters.t;
    const near = { w: parameters.near, q: 10 };
    const far = { w: parameters.far, q: 40 };
    const inverseW = (1 - t) / near.w + t / far.w;
    return {
      t,
      linear: {
        w: (1 - t) * near.w + t * far.w,
        q: (1 - t) * near.q + t * far.q,
      },
      correct: {
        inverseW,
        w: 1 / inverseW,
        q: ((1 - t) * near.q / near.w + t * far.q / far.w) / inverseW,
      },
      depths,
      order: orderSelect.value,
    };
  }

  function syncOutputs() {
    [...interpolationInputs, ...depthInputs].forEach((input) => {
      input.nextElementSibling.value = fmt(Number(input.value));
    });
  }

  function drawInterpolation(canvas, context, result, title, color, correct) {
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#10242c";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#9db6b8";
    context.font = "14px system-ui";
    context.fillText(title, 20, 30);
    context.fillStyle = "#ef9a67";
    context.fillText("近端：w=1，q=10", 20, 64);
    context.fillStyle = "#58c7bd";
    context.fillText(`远端：w=${fmt(window.RTBook.clamp(Number(mount.querySelector('[data-interp="far"]').value), 2, 10))}，q=40`, 20, 88);

    const barX = 24;
    const barY = 122;
    const barWidth = canvas.width - 48;
    context.fillStyle = "rgba(255,255,255,.12)";
    context.fillRect(barX, barY, barWidth, 28);
    context.fillStyle = color;
    context.fillRect(barX, barY, barWidth * window.RTBook.clamp(result.q / 40, 0, 1), 28);
    context.strokeStyle = "rgba(255,255,255,.28)";
    context.strokeRect(barX + 0.5, barY + 0.5, barWidth - 1, 27);

    const markerX = barX + result.t * barWidth;
    context.fillStyle = "#ffd48a";
    context.beginPath();
    context.arc(markerX, barY + 14, 6, 0, Math.PI * 2);
    context.fill();
    context.fillStyle = "#f6f3eb";
    context.font = "16px system-ui";
    context.fillText(`t=${fmt(result.t)}`, 24, 184);
    context.fillText(`w=${fmt(result.w)}`, 24, 212);
    context.fillText(`q=${fmt(result.q)}`, 170, 212);
    if (correct) {
      context.fillStyle = "#9db6b8";
      context.font = "13px system-ui";
      context.fillText(`1/w=${fmt(result.inverseW)}`, 170, 184);
    }
  }

  function drawOverlap(context, x, y, first, winner) {
    const colors = { A: "#c66b35", B: "#087f83" };
    const draw = (id) => {
      context.fillStyle = colors[id];
      context.fillRect(x + (id === "A" ? 0 : 44), y + (id === "A" ? 28 : 0), 112, 88);
      context.fillStyle = "#f6f3eb";
      context.font = "bold 18px system-ui";
      context.fillText(id, x + (id === "A" ? 10 : 54) + 42, y + (id === "A" ? 28 : 0) + 48);
    };
    draw(first);
    draw(first === "A" ? "B" : "A");
    context.strokeStyle = "#ffd48a";
    context.lineWidth = 2;
    context.strokeRect(x + 70, y + 46, 34, 34);
    context.lineWidth = 1;
    context.fillStyle = "#ffd48a";
    context.font = "13px system-ui";
    context.fillText(`最终写入 ${winner}`, x + 6, y + 142);
  }

  function drawOrder(result) {
    const depths = result.depths;
    const painter = result.order === "AB" ? "B" : "A";
    const first = result.order[0];
    const nearest = depths.a <= depths.b ? "A" : "B";
    orderContext.clearRect(0, 0, orderCanvas.width, orderCanvas.height);
    orderContext.fillStyle = "#10242c";
    orderContext.fillRect(0, 0, orderCanvas.width, orderCanvas.height);
    orderContext.fillStyle = "#9db6b8";
    orderContext.font = "15px system-ui";
    orderContext.fillText(`画家算法 · 提交顺序 ${result.order}`, 24, 32);
    orderContext.fillText("z-buffer · 与提交顺序无关", 340, 32);
    orderContext.strokeStyle = "rgba(255,255,255,.18)";
    orderContext.beginPath();
    orderContext.moveTo(320, 12);
    orderContext.lineTo(320, 268);
    orderContext.stroke();
    drawOverlap(orderContext, 58, 62, first, painter);
    drawOverlap(orderContext, 354, 62, nearest === "A" ? "B" : "A", nearest);
    orderContext.fillStyle = "#f6f3eb";
    orderContext.font = "14px system-ui";
    orderContext.fillText(`同一像素：A 深度=${fmt(depths.a)} · B 深度=${fmt(depths.b)}`, 24, 252);
    orderContext.fillStyle = "#ffd48a";
    orderContext.fillText(`z-buffer 保留 ${nearest}`, 434, 252);
    return { painter, nearest };
  }

  function update() {
    syncOutputs();
    const result = values();
    drawInterpolation(linearCanvas, linearContext, result.linear, "直接线性插值（错误）", "#ef9a67", false);
    drawInterpolation(inverseCanvas, inverseContext, result.correct, "透视正确插值", "#58c7bd", true);
    const order = drawOrder(result);
    readout.textContent =
      `t=${fmt(result.t)} · 线性 w=${fmt(result.linear.w)} / q=${fmt(result.linear.q)} · ` +
      `正确 w=${fmt(result.correct.w)} / q=${fmt(result.correct.q)} · 1/w=${fmt(result.correct.inverseW)} · ` +
      `A=${fmt(result.depths.a)} · B=${fmt(result.depths.b)} · 画家(${result.order})=${order.painter} · z-buffer=${order.nearest}`;
  }

  interpolationInputs.forEach((input) => {
    input.addEventListener("input", () => {
      const key = input.dataset.interp;
      const nearInput = interpolationInputs.find((item) => item.dataset.interp === "near");
      const farInput = interpolationInputs.find((item) => item.dataset.interp === "far");
      if (key === "near" && Number(nearInput.value) >= Number(farInput.value)) {
        farInput.value = String(Math.min(10, Number(nearInput.value) + 1));
      }
      if (key === "far" && Number(farInput.value) <= Number(nearInput.value)) {
        nearInput.value = String(Math.max(1, Number(farInput.value) - 1));
      }
      update();
    });
  });
  depthInputs.forEach((input) => input.addEventListener("input", update));
  orderSelect.addEventListener("change", update);
  mount.querySelector("[data-reset]").addEventListener("click", () => {
    const interpolationDefaults = { t: 0.5, near: 1, far: 4 };
    interpolationInputs.forEach((input) => { input.value = String(interpolationDefaults[input.dataset.interp]); });
    const depthDefaults = { a: 3, b: 5 };
    depthInputs.forEach((input) => { input.value = String(depthDefaults[input.dataset.depth]); });
    orderSelect.value = "AB";
    update();
  });

  update();
})();
