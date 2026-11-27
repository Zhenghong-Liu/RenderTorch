(() => {
  const mount = document.getElementById("shadow-demo");
  if (!mount) return;

  mount.insertAdjacentHTML(
    "beforeend",
    `<div class="demo-toolbar">
      <label>光源方位角 <input type="range" data-shadow="azimuth" min="-180" max="180" step="1" value="-35"><output>-35°</output></label>
      <label>光源仰角 <input type="range" data-shadow="elevation" min="15" max="80" step="1" value="42"><output>42°</output></label>
      <label>depth bias <input type="range" data-shadow="bias" min="0" max="0.02" step="0.0001" value="0.0012"><output>0.0012</output></label>
      <label>exposure <input type="range" data-shadow="exposure" min="0.4" max="2" step="0.05" value="1.1"><output>1.10</output></label>
      <label>gamma <select data-gamma><option value="2.2" selected>2.2</option><option value="1.0">1.0（关闭）</option></select></label>
      <button type="button" data-reset>重置</button>
    </div>
    <div class="demo-grid">
      <figure><canvas id="shadow-map-canvas" width="360" height="360" role="img" aria-label="光源正交视图生成的 48 乘 48 CPU 阴影深度图"></canvas><figcaption>光源正交视图 · 48 × 48 深度数组</figcaption></figure>
      <figure><canvas id="shadow-scene-canvas" width="540" height="360" role="img" aria-label="方块与球在地面上的实时 CPU 阴影"></canvas><figcaption>最终场景 · 地面探针随阴影同步更新</figcaption></figure>
    </div>
    <figure><canvas id="shadow-color-canvas" width="820" height="154" role="img" aria-label="地面探针的线性颜色、gamma 后颜色与 Reinhard 后颜色"></canvas><figcaption>同一线性颜色经 gamma 与 Reinhard 色调映射后的显示值</figcaption></figure>
    <p class="demo-readout" data-readout aria-live="polite"></p>`
  );

  const mapCanvas = mount.querySelector("#shadow-map-canvas");
  const sceneCanvas = mount.querySelector("#shadow-scene-canvas");
  const colorCanvas = mount.querySelector("#shadow-color-canvas");
  const mapContext = mapCanvas.getContext("2d");
  const sceneContext = sceneCanvas.getContext("2d");
  const colorContext = colorCanvas.getContext("2d");
  const inputs = [...mount.querySelectorAll("[data-shadow]")];
  const gammaSelect = mount.querySelector("[data-gamma]");
  const readout = mount.querySelector("[data-readout]");
  const defaults = { azimuth: -35, elevation: 42, bias: 0.0012, exposure: 1.1 };
  const mapSize = 48;
  const mapImage = mapContext.createImageData(mapSize, mapSize);
  const mapBuffer = document.createElement("canvas");
  mapBuffer.width = mapSize;
  mapBuffer.height = mapSize;
  const mapBufferContext = mapBuffer.getContext("2d");
  const depthMap = new Float32Array(mapSize * mapSize);
  const near = 0.1;
  const far = 20;
  const lightDistance = 5;
  const extent = 4.4;
  const target = { x: 0, y: 0.8, z: 0 };
  const cube = { min: { x: -1.25, y: 0, z: -0.55 }, max: { x: 0.2, y: 1.5, z: 0.9 } };
  const sphere = { center: { x: 0.95, y: 0.74, z: -0.78 }, radius: 0.74 };
  const probe = { x: 0, y: 0.008, z: 1 };
  const camera = { position: { x: 5.4, y: 4.3, z: 7.2 }, target: { x: 0, y: 0.68, z: 0 }, fov: 0.72 };
  const fmt = (value, digits = 3) => {
    const safe = Number.isFinite(value) ? value : 0;
    if (window.RTBook && typeof window.RTBook.formatNumber === "function") return window.RTBook.formatNumber(safe, digits);
    return safe.toFixed(digits);
  };
  const clamp = (value, low, high) => Math.min(high, Math.max(low, Number.isFinite(value) ? value : low));
  const add = (a, b) => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z });
  const sub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
  const scale = (a, amount) => ({ x: a.x * amount, y: a.y * amount, z: a.z * amount });
  const dot = (a, b) => a.x * b.x + a.y * b.y + a.z * b.z;
  const cross = (a, b) => ({ x: a.y * b.z - a.z * b.y, y: a.z * b.x - a.x * b.z, z: a.x * b.y - a.y * b.x });
  const length = (a) => Math.hypot(a.x, a.y, a.z);
  const normalize = (a) => {
    const magnitude = length(a);
    return magnitude > 1e-12 ? scale(a, 1 / magnitude) : { x: 0, y: 0, z: 0 };
  };

  function parameters() {
    const raw = Object.fromEntries(inputs.map((input) => [input.dataset.shadow, Number(input.value)]));
    const state = {
      azimuth: clamp(raw.azimuth, -180, 180),
      elevation: clamp(raw.elevation, 15, 80),
      bias: clamp(raw.bias, 0, 0.02),
      exposure: clamp(raw.exposure, 0.4, 2),
    };
    const azimuth = state.azimuth * Math.PI / 180;
    const elevation = state.elevation * Math.PI / 180;
    const light = {
      x: Math.cos(elevation) * Math.cos(azimuth),
      y: Math.sin(elevation),
      z: Math.cos(elevation) * Math.sin(azimuth),
    };
    return { ...state, light, rayDirection: scale(light, -1), gamma: clamp(Number(gammaSelect.value), 1, 2.2) };
  }

  function lightBasis(light) {
    const right = normalize(cross({ x: 0, y: 1, z: 0 }, light));
    return { right, up: normalize(cross(light, right)) };
  }

  function rayBox(origin, direction, box) {
    let low = 0;
    let high = Infinity;
    const axes = ["x", "y", "z"];
    for (const axis of axes) {
      if (Math.abs(direction[axis]) < 1e-12) {
        if (origin[axis] < box.min[axis] || origin[axis] > box.max[axis]) return null;
        continue;
      }
      let first = (box.min[axis] - origin[axis]) / direction[axis];
      let second = (box.max[axis] - origin[axis]) / direction[axis];
      if (first > second) [first, second] = [second, first];
      low = Math.max(low, first);
      high = Math.min(high, second);
      if (high < low) return null;
    }
    if (!Number.isFinite(high) || high < 0) return null;
    const distance = low >= 0 ? low : high;
    const point = add(origin, scale(direction, distance));
    const eps = 1e-5;
    let normal = { x: 0, y: 0, z: 0 };
    if (Math.abs(point.x - box.min.x) < eps) normal.x = -1;
    else if (Math.abs(point.x - box.max.x) < eps) normal.x = 1;
    else if (Math.abs(point.y - box.min.y) < eps) normal.y = -1;
    else if (Math.abs(point.y - box.max.y) < eps) normal.y = 1;
    else if (Math.abs(point.z - box.min.z) < eps) normal.z = -1;
    else if (Math.abs(point.z - box.max.z) < eps) normal.z = 1;
    return { distance, point, normal, type: "cube" };
  }

  function raySphere(origin, direction, object) {
    const offset = sub(origin, object.center);
    const b = dot(offset, direction);
    const c = dot(offset, offset) - object.radius * object.radius;
    const discriminant = b * b - c;
    if (discriminant < 0) return null;
    const root = Math.sqrt(discriminant);
    const candidates = [-b - root, -b + root];
    const distance = candidates.find((value) => value > 1e-5);
    if (!Number.isFinite(distance)) return null;
    const point = add(origin, scale(direction, distance));
    return { distance, point, normal: normalize(sub(point, object.center)), type: "sphere" };
  }

  function nearestObject(origin, direction) {
    const boxHit = rayBox(origin, direction, cube);
    const sphereHit = raySphere(origin, direction, sphere);
    if (!boxHit) return sphereHit;
    if (!sphereHit) return boxHit;
    return boxHit.distance <= sphereHit.distance ? boxHit : sphereHit;
  }

  function projectToLight(point, state) {
    const relative = sub(point, target);
    const basis = lightBasis(state.light);
    const u = dot(relative, basis.right) / (extent * 2) + 0.5;
    const v = 0.5 - dot(relative, basis.up) / (extent * 2);
    const distance = lightDistance + near - dot(relative, state.light);
    return { u, v, depth: (distance - near) / (far - near), distance };
  }

  function buildShadowMap(state) {
    depthMap.fill(Infinity);
    const basis = lightBasis(state.light);
    let writes = 0;
    for (let y = 0; y < mapSize; y += 1) {
      for (let x = 0; x < mapSize; x += 1) {
        const u = (x + 0.5) / mapSize * 2 - 1;
        const v = 1 - (y + 0.5) / mapSize * 2;
        const origin = add(add(add(target, scale(basis.right, u * extent)), scale(basis.up, v * extent)), scale(state.light, lightDistance + near));
        const hit = nearestObject(origin, state.rayDirection);
        if (!hit || hit.distance < near || hit.distance > far) continue;
        const depth = (hit.distance - near) / (far - near);
        if (Number.isFinite(depth) && depth >= 0 && depth <= 1) {
          depthMap[y * mapSize + x] = depth;
          writes += 1;
        }
      }
    }
    return writes;
  }

  function visibility(point, normal, state) {
    const offsetPoint = add(point, scale(normal, 0.004));
    const projected = projectToLight(offsetPoint, state);
    const x = Math.floor(projected.u * mapSize);
    const y = Math.floor(projected.v * mapSize);
    if (x < 0 || x >= mapSize || y < 0 || y >= mapSize) return { visible: true, stored: Infinity, current: projected.depth, difference: Infinity, x, y };
    const stored = depthMap[y * mapSize + x];
    if (!Number.isFinite(stored)) return { visible: true, stored, current: projected.depth, difference: Infinity, x, y };
    const current = projected.depth;
    const difference = current - stored;
    return { visible: stored + state.bias >= current, stored, current, difference, x, y };
  }

  function drawShadowMap(state, writes) {
    for (let i = 0; i < depthMap.length; i += 1) {
      const depth = depthMap[i];
      const shade = Number.isFinite(depth) ? Math.round(55 + (1 - depth) * 200) : 12;
      const offset = i * 4;
      mapImage.data[offset] = Number.isFinite(depth) ? shade : 18;
      mapImage.data[offset + 1] = Number.isFinite(depth) ? Math.min(255, shade + 18) : 31;
      mapImage.data[offset + 2] = Number.isFinite(depth) ? Math.min(255, shade + 22) : 37;
      mapImage.data[offset + 3] = 255;
    }
    mapContext.clearRect(0, 0, mapCanvas.width, mapCanvas.height);
    mapContext.fillStyle = "#10242c";
    mapContext.fillRect(0, 0, mapCanvas.width, mapCanvas.height);
    mapContext.fillStyle = "#f6f3eb";
    mapContext.font = "bold 15px system-ui";
    mapContext.fillText("CPU shadow map", 18, 29);
    mapContext.fillStyle = "#9db6b8";
    mapContext.font = "12px system-ui";
    mapContext.fillText("近 = 亮　远 = 暗　空白 = ∞", 18, 49);
    mapContext.imageSmoothingEnabled = false;
    mapBufferContext.putImageData(mapImage, 0, 0);
    mapContext.drawImage(mapBuffer, 0, 0, mapSize, mapSize, 56, 72, 248, 248);
    mapContext.strokeStyle = "rgba(255,255,255,.25)";
    mapContext.strokeRect(55.5, 71.5, 249, 249);
    mapContext.fillStyle = "#58c7bd";
    mapContext.font = "13px system-ui";
    mapContext.fillText(`写入 ${writes} / ${mapSize * mapSize}`, 56, 344);
  }

  function cameraBasis() {
    const forward = normalize(sub(camera.target, camera.position));
    const right = normalize(cross(forward, { x: 0, y: 1, z: 0 }));
    return { forward, right, up: cross(right, forward) };
  }

  function surfaceColor(hit, state) {
    const lightAmount = Math.max(0, dot(hit.normal, state.light));
    const albedo = hit.type === "cube" ? { x: 0.7, y: 0.29, z: 0.12 } : { x: 0.1, y: 0.52, z: 0.58 };
    if (hit.type === "cube") {
      const local = sub(hit.point, { x: -0.525, y: 0.75, z: 0.175 });
      if (Math.abs(hit.normal.y) < 0.5) albedo.x += 0.08 * Math.sin(local.y * 15);
    }
    const shadow = visibility(hit.point, hit.normal, state).visible ? 1 : 0;
    const direct = lightAmount * shadow;
    const ambient = 0.16;
    const energy = ambient + direct * 0.92;
    return { x: albedo.x * energy, y: albedo.y * energy, z: albedo.z * energy };
  }

  function drawScene(state) {
    const width = sceneCanvas.width;
    const height = sceneCanvas.height;
    const image = sceneContext.createImageData(width, height);
    const basis = cameraBasis();
    const aspect = width / height;
    const tangent = Math.tan(camera.fov / 2);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const sx = (2 * (x + 0.5) / width - 1) * aspect * tangent;
        const sy = (1 - 2 * (y + 0.5) / height) * tangent;
        const direction = normalize(add(add(basis.forward, scale(basis.right, sx)), scale(basis.up, sy)));
        const objectHit = nearestObject(camera.position, direction);
        const planeDistance = Math.abs(direction.y) > 1e-8 ? -camera.position.y / direction.y : Infinity;
        const planeValid = planeDistance > 0 && (!objectHit || planeDistance < objectHit.distance);
        let linear;
        let hit;
        if (planeValid) {
          hit = {
            type: "ground",
            point: add(camera.position, scale(direction, planeDistance)),
            normal: { x: 0, y: 1, z: 0 },
          };
          const lightAmount = state.light.y;
          const shadow = visibility(hit.point, hit.normal, state).visible ? 1 : 0;
          const checker = (Math.floor(hit.point.x * 1.4) + Math.floor(hit.point.z * 1.4)) % 2 === 0 ? 0.26 : 0.3;
          const fade = 0.45 + 0.55 * Math.max(0, 1 - length(hit.point) / 7);
          const energy = (0.16 + lightAmount * shadow * 0.92) * fade;
          linear = { x: checker * energy * 0.86, y: checker * energy, z: checker * energy * 0.8 };
        } else if (objectHit) {
          hit = objectHit;
          linear = surfaceColor(hit, state);
        } else {
          const gradient = clamp(y / height, 0, 1);
          linear = { x: 0.045 + gradient * 0.025, y: 0.075 + gradient * 0.025, z: 0.09 + gradient * 0.03 };
        }
        const offset = (y * width + x) * 4;
        image.data[offset] = Math.round(clamp(Math.pow(linear.x * state.exposure, 1 / state.gamma), 0, 1) * 255);
        image.data[offset + 1] = Math.round(clamp(Math.pow(linear.y * state.exposure, 1 / state.gamma), 0, 1) * 255);
        image.data[offset + 2] = Math.round(clamp(Math.pow(linear.z * state.exposure, 1 / state.gamma), 0, 1) * 255);
        image.data[offset + 3] = 255;
      }
    }
    sceneContext.putImageData(image, 0, 0);
    const screen = projectScreen(probe, basis, width, height);
    sceneContext.strokeStyle = visibility(probe, { x: 0, y: 1, z: 0 }, state).visible ? "#ffd48a" : "#ef7f74";
    sceneContext.lineWidth = 2;
    sceneContext.beginPath();
    sceneContext.arc(screen.x, screen.y, 9, 0, Math.PI * 2);
    sceneContext.moveTo(screen.x - 14, screen.y);
    sceneContext.lineTo(screen.x + 14, screen.y);
    sceneContext.moveTo(screen.x, screen.y - 14);
    sceneContext.lineTo(screen.x, screen.y + 14);
    sceneContext.stroke();
    sceneContext.lineWidth = 1;
    sceneContext.fillStyle = "#f6f3eb";
    sceneContext.font = "13px system-ui";
    sceneContext.fillText("探针", screen.x + 17, screen.y - 12);
  }

  function projectScreen(point, basis, width, height) {
    const relative = sub(point, camera.position);
    const depth = Math.max(1e-8, dot(relative, basis.forward));
    const aspect = width / height;
    const tangent = Math.tan(camera.fov / 2);
    return {
      x: (0.5 + dot(relative, basis.right) / (depth * aspect * tangent) * 0.5) * width,
      y: (0.5 - dot(relative, basis.up) / (depth * tangent) * 0.5) * height,
    };
  }

  function colorBytes(linear) {
    return [linear.x, linear.y, linear.z].map((value) => Math.round(clamp(Math.pow(value, 1 / 2.2), 0, 1) * 255));
  }

  function reinhard(linear) {
    return [linear.x, linear.y, linear.z].map((value) => {
      const denominator = 1 + value;
      return Number.isFinite(denominator) && Math.abs(denominator) > 1e-12 ? value / denominator : 0;
    });
  }

  function drawColors(linear, state) {
    const gamma = [linear.x, linear.y, linear.z].map((value) => Math.pow(clamp(value, 0, 1), 1 / state.gamma));
    const mapped = reinhard(linear);
    const items = [
      { title: "C_linear", formula: "光照结果（线性）", values: [linear.x, linear.y, linear.z] },
      { title: "C_linear^(1/2.2)", formula: `固定 gamma 2.2 · 当前 ${fmt(state.gamma, 1)}`, values: [Math.pow(clamp(linear.x, 0, 1), 1 / 2.2), Math.pow(clamp(linear.y, 0, 1), 1 / 2.2), Math.pow(clamp(linear.z, 0, 1), 1 / 2.2)] },
      { title: "C_reinhard", formula: "C_linear / (1 + C_linear)", values: mapped },
    ];
    colorContext.clearRect(0, 0, colorCanvas.width, colorCanvas.height);
    colorContext.fillStyle = "#10242c";
    colorContext.fillRect(0, 0, colorCanvas.width, colorCanvas.height);
    items.forEach((item, index) => {
      const x = 20 + index * 268;
      colorContext.fillStyle = `rgb(${colorBytes(item.values).join(",")})`;
      colorContext.fillRect(x, 18, 70, 70);
      colorContext.strokeStyle = "rgba(255,255,255,.35)";
      colorContext.strokeRect(x + 0.5, 18.5, 69, 69);
      colorContext.fillStyle = "#f6f3eb";
      colorContext.font = "bold 14px system-ui";
      colorContext.fillText(item.title, x + 82, 37);
      colorContext.fillStyle = "#9db6b8";
      colorContext.font = "11px system-ui";
      colorContext.fillText(item.formula, x + 82, 57);
      colorContext.fillStyle = "#58c7bd";
      colorContext.font = "13px system-ui";
      colorContext.fillText(`(${item.values.map((value) => fmt(value)).join(", ")})`, x, 117);
      if (index < 2) {
        colorContext.strokeStyle = "rgba(255,255,255,.14)";
        colorContext.beginPath();
        colorContext.moveTo(x + 248, 18);
        colorContext.lineTo(x + 248, 128);
        colorContext.stroke();
      }
    });
    colorContext.fillStyle = "#9db6b8";
    colorContext.font = "11px system-ui";
    colorContext.fillText("色块使用 sRGB 显示；线性色块仅作数值对照。", 20, 143);
  }

  function syncOutputs() {
    inputs.forEach((input) => {
      const value = Number(input.value);
      input.nextElementSibling.value = input.dataset.shadow === "bias" ? fmt(value, 4) : input.dataset.shadow === "azimuth" || input.dataset.shadow === "elevation" ? `${fmt(value, 0)}°` : fmt(value, 2);
    });
  }

  function update() {
    syncOutputs();
    const state = parameters();
    const writes = buildShadowMap(state);
    const probeResult = visibility(probe, { x: 0, y: 1, z: 0 }, state);
    drawShadowMap(state, writes);
    drawScene(state);
    const checker = (Math.floor(probe.x * 1.4) + Math.floor(probe.z * 1.4)) % 2 === 0 ? 0.26 : 0.3;
    const fade = 0.45 + 0.55 * Math.max(0, 1 - length(probe) / 7);
    const lightAmount = probeResult.visible ? state.light.y : 0;
    const energy = (0.16 + lightAmount * 0.92) * fade;
    const linear = { x: checker * energy * 0.86, y: checker * energy, z: checker * energy * 0.8 };
    const exposed = { x: linear.x * state.exposure, y: linear.y * state.exposure, z: linear.z * state.exposure };
    drawColors(exposed, state);
    const mapIndex = probeResult.x >= 0 && probeResult.x < mapSize && probeResult.y >= 0 && probeResult.y < mapSize ? probeResult.y * mapSize + probeResult.x : -1;
    const difference = Number.isFinite(probeResult.difference) ? probeResult.difference : 0;
    readout.textContent =
      `方位 ${fmt(state.azimuth, 0)}° · 仰角 ${fmt(state.elevation, 0)}° · bias ${fmt(state.bias, 4)} · exposure ${fmt(state.exposure, 2)} · ` +
      `shadow map 写入 ${writes}/${mapSize * mapSize} 样本 · 地面探针 (${fmt(probe.x, 2)}, ${fmt(probe.z, 2)}) ` +
      `${probeResult.visible ? "可见（未遮挡）" : "遮挡（阴影）"} · 探针纹素 [${probeResult.x}, ${probeResult.y}] · ` +
      `${Number.isFinite(probeResult.stored) ? `stored ${fmt(probeResult.stored, 5)} / current ${fmt(probeResult.current, 5)} / Δ ${fmt(difference, 5)}` : "该纹素为空"} · ` +
      `线性 RGB (${[exposed.x, exposed.y, exposed.z].map((value) => fmt(value)).join(", ")})`;
    if (mapIndex >= 0) mapCanvas.dataset.probeIndex = String(mapIndex);
  }

  inputs.forEach((input) => input.addEventListener("input", update));
  gammaSelect.addEventListener("change", update);
  mount.querySelector("[data-reset]").addEventListener("click", () => {
    inputs.forEach((input) => { input.value = String(defaults[input.dataset.shadow]); });
    gammaSelect.value = "2.2";
    update();
  });

  update();
})();
