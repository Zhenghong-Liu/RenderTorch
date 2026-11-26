(() => {
  const mount = document.getElementById("raytrace-demo");
  if (!mount) return;

  mount.insertAdjacentHTML(
    "beforeend",
    `<div class="demo-toolbar">
      <label>最大反射 <input type="range" data-ray="bounces" min="0" max="3" step="1" value="2"><output>2</output></label>
      <label>每像素样本 <input type="range" data-ray="samples" min="1" max="4" step="1" value="2"><output>2</output></label>
      <label>球心 x <input type="range" data-ray="sphereX" min="-2" max="2" step="0.1" value="0"><output>0.0</output></label>
      <label>相机角度 <input type="range" data-ray="angle" min="-30" max="30" step="1" value="-12"><output>-12°</output></label>
      <button type="button" data-reset>重置</button>
    </div>
    <canvas id="raytrace-canvas" width="480" height="300" role="img" aria-label="CPU 光线追踪的球体、镜面反射与棋盘地面" style="display:block;width:100%;height:auto"></canvas>
    <p class="demo-readout" data-readout aria-live="polite"></p>`
  );

  const canvas = mount.querySelector("#raytrace-canvas");
  const context = canvas.getContext("2d");
  if (!context) return;
  const inputs = [...mount.querySelectorAll("[data-ray]")];
  const readout = mount.querySelector("[data-readout]");
  const resetButton = mount.querySelector("[data-reset]");
  const defaults = { bounces: 2, samples: 2, sphereX: 0, angle: -12 };
  const EPS = 1e-5;
  const light = normalize([-0.45, 0.82, -0.36]);
  const sampleOffsets = [
    [[0, 0]],
    [[-0.25, 0], [0.25, 0]],
    [[-0.25, -0.25], [0.25, -0.25], [0, 0.25]],
    [[-0.25, -0.25], [0.25, -0.25], [-0.25, 0.25], [0.25, 0.25]],
  ];
  const fmt = (value) => (Number.isFinite(value) ? value : 0).toFixed(3);
  const vectorText = (vector) => vector ? `(${fmt(vector[0])}, ${fmt(vector[1])}, ${fmt(vector[2])})` : "—";

  function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  function normalize(vector) {
    const length = Math.hypot(vector[0], vector[1], vector[2]);
    if (!Number.isFinite(length) || length <= EPS) return [0, 0, 0];
    return [vector[0] / length, vector[1] / length, vector[2] / length];
  }

  function values() {
    const result = {};
    inputs.forEach((input) => {
      const value = Number(input.value);
      result[input.dataset.ray] = Number.isFinite(value) ? value : defaults[input.dataset.ray];
    });
    result.bounces = Math.max(0, Math.min(3, Math.round(result.bounces)));
    result.samples = Math.max(1, Math.min(4, Math.round(result.samples)));
    result.sphereX = Math.max(-2, Math.min(2, result.sphereX));
    result.angle = Math.max(-30, Math.min(30, result.angle));
    return result;
  }

  function syncOutputs() {
    inputs.forEach((input) => {
      const value = Number(input.value);
      input.nextElementSibling.value = input.dataset.ray === "angle" ? `${Math.round(value)}°` : input.dataset.ray === "sphereX" ? value.toFixed(1) : String(Math.round(value));
    });
  }

  function intersectSphere(origin, direction, sphere) {
    const oc = [origin[0] - sphere.center[0], origin[1] - sphere.center[1], origin[2] - sphere.center[2]];
    const a = dot(direction, direction);
    const b = 2 * dot(oc, direction);
    const c = dot(oc, oc) - sphere.radius * sphere.radius;
    if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(c) || a <= EPS || sphere.radius <= 0) return null;
    const discriminant = b * b - 4 * a * c;
    if (!Number.isFinite(discriminant) || discriminant < 0) return null;
    const root = Math.sqrt(discriminant);
    const near = (-b - root) / (2 * a);
    const far = (-b + root) / (2 * a);
    const t = near >= 0 ? near : far >= 0 ? far : NaN;
    if (!Number.isFinite(t) || t < 0) return null;
    const point = [origin[0] + t * direction[0], origin[1] + t * direction[1], origin[2] + t * direction[2]];
    const normal = normalize([point[0] - sphere.center[0], point[1] - sphere.center[1], point[2] - sphere.center[2]]);
    if (dot(normal, normal) <= EPS) return null;
    return { name: "球体", t, point, normal, color: [0.24, 0.29, 0.35], reflectivity: 0.82 };
  }

  function intersectPlane(origin, direction, point, normal) {
    const denominator = dot(direction, normal);
    if (!Number.isFinite(denominator) || Math.abs(denominator) <= EPS) return null;
    const offset = [point[0] - origin[0], point[1] - origin[1], point[2] - origin[2]];
    const t = dot(offset, normal) / denominator;
    if (!Number.isFinite(t) || t < 0) return null;
    const hitPoint = [origin[0] + t * direction[0], origin[1] + t * direction[1], origin[2] + t * direction[2]];
    return {
      name: "地面",
      t,
      point: hitPoint,
      normal: denominator < 0 ? [...normal] : [-normal[0], -normal[1], -normal[2]],
      color: (Math.floor(hitPoint[0] * 1.4) + Math.floor(hitPoint[2] * 1.4)) % 2 === 0 ? [0.29, 0.36, 0.38] : [0.13, 0.18, 0.21],
      reflectivity: 0.2,
    };
  }

  function reflect(direction, normal) {
    if (!direction.every(Number.isFinite) || !normal.every(Number.isFinite)) return [0, 0, 0];
    const projection = 2 * dot(direction, normal);
    return [direction[0] - projection * normal[0], direction[1] - projection * normal[1], direction[2] - projection * normal[2]];
  }

  function shade(hit) {
    const diffuse = Math.max(0, dot(hit.normal, light));
    return hit.color.map((channel) => channel * (0.16 + 0.84 * diffuse));
  }

  function environment(direction) {
    const sky = Math.max(0, Math.min(1, direction[1] * 0.5 + 0.5));
    const color = [0.08 + sky * 0.16, 0.12 + sky * 0.25, 0.17 + sky * 0.33];
    const card = Math.max(0, dot(direction, normalize([-0.55, 0.65, -0.52])));
    const glow = Math.pow(card, 80) * 0.75;
    return [color[0] + glow, color[1] + glow * 0.82, color[2] + glow * 0.55];
  }

  function nearestHit(origin, direction, sphere) {
    const sphereHit = intersectSphere(origin, direction, sphere);
    const planeHit = intersectPlane(origin, direction, [0, 0, 0], [0, 1, 0]);
    if (!sphereHit) return planeHit;
    if (!planeHit) return sphereHit;
    return sphereHit.t <= planeHit.t ? sphereHit : planeHit;
  }

  function trace(origin, direction, sphere, maxBounces, record = false) {
    let rayOrigin = [...origin];
    let rayDirection = normalize(direction);
    let throughput = [1, 1, 1];
    const color = [0, 0, 0];
    const result = { color, bounces: 0, origin: [...origin], direction: [...rayDirection], hit: null, reflection: null };
    for (let bounce = 0; bounce <= maxBounces; bounce += 1) {
      const hit = nearestHit(rayOrigin, rayDirection, sphere);
      if (!hit) {
        const background = environment(rayDirection);
        for (let channel = 0; channel < 3; channel += 1) color[channel] += throughput[channel] * background[channel];
        result.bounces = bounce;
        break;
      }
      if (record && bounce === 0) result.hit = hit;
      const direct = shade(hit);
      for (let channel = 0; channel < 3; channel += 1) color[channel] += throughput[channel] * direct[channel];
      if (bounce === maxBounces) {
        result.bounces = bounce;
        break;
      }
      const reflection = normalize(reflect(rayDirection, hit.normal));
      if (dot(reflection, reflection) <= EPS) {
        result.bounces = bounce;
        break;
      }
      if (record && bounce === 0) result.reflection = reflection;
      throughput = throughput.map((channel, index) => channel * hit.reflectivity);
      rayOrigin = [hit.point[0] + hit.normal[0] * EPS, hit.point[1] + hit.normal[1] * EPS, hit.point[2] + hit.normal[2] * EPS];
      rayDirection = reflection;
      result.bounces = bounce + 1;
    }
    result.rgb = result.color.map((channel) => Math.round(255 * Math.max(0, Math.min(1, Number.isFinite(channel) ? channel : 0))));
    return result;
  }

  function camera(parameters) {
    const angle = parameters.angle * Math.PI / 180;
    const origin = [Math.sin(angle) * 6, 2.35, Math.cos(angle) * 6];
    const forward = normalize([-origin[0], 1 - origin[1], 3 - origin[2]]);
    const right = normalize([-forward[2], 0, forward[0]]);
    const up = normalize([right[1] * forward[2] - right[2] * forward[1], right[2] * forward[0] - right[0] * forward[2], right[0] * forward[1] - right[1] * forward[0]]);
    return { origin, forward, right, up };
  }

  function render() {
    const parameters = values();
    syncOutputs();
    const sphere = { center: [parameters.sphereX, 1, 3], radius: 1 };
    const view = camera(parameters);
    const aspect = canvas.width / canvas.height;
    const scale = Math.tan(42 * Math.PI / 360);
    const offsets = sampleOffsets[parameters.samples - 1];
    const image = context.createImageData(canvas.width, canvas.height);
    let centerTrace = null;
    for (let y = 0; y < canvas.height; y += 1) {
      for (let x = 0; x < canvas.width; x += 1) {
        const total = [0, 0, 0];
        offsets.forEach(([offsetX, offsetY]) => {
          const u = (2 * (x + 0.5 + offsetX) / canvas.width - 1) * aspect * scale;
          const v = (1 - 2 * (y + 0.5 + offsetY) / canvas.height) * scale;
          const direction = normalize([
            view.forward[0] + view.right[0] * u + view.up[0] * v,
            view.forward[1] + view.right[1] * u + view.up[1] * v,
            view.forward[2] + view.right[2] * u + view.up[2] * v,
          ]);
          const traced = trace(view.origin, direction, sphere, parameters.bounces);
          for (let channel = 0; channel < 3; channel += 1) total[channel] += traced.rgb[channel];
        });
        const pixel = (y * canvas.width + x) * 4;
        for (let channel = 0; channel < 3; channel += 1) image.data[pixel + channel] = Math.round(total[channel] / offsets.length);
        image.data[pixel + 3] = 255;
      }
    }
    context.putImageData(image, 0, 0);
    centerTrace = trace(view.origin, [
      view.forward[0],
      view.forward[1],
      view.forward[2],
    ], sphere, parameters.bounces, true);
    readout.textContent =
      `代表光线（画布中心） · origin=${vectorText(centerTrace.origin)} · direction=${vectorText(centerTrace.direction)} · ` +
      `命中=${centerTrace.hit ? centerTrace.hit.name : "无"} · t=${centerTrace.hit ? fmt(centerTrace.hit.t) : "—"} · ` +
      `法线=${vectorText(centerTrace.hit ? centerTrace.hit.normal : null)} · 反射=${vectorText(centerTrace.reflection)} · ` +
      `bounce=${centerTrace.bounces} · 最大反射=${parameters.bounces} · RGB=${vectorText(centerTrace.rgb)}`;
  }

  inputs.forEach((input) => input.addEventListener("input", render));
  resetButton.addEventListener("click", () => {
    inputs.forEach((input) => { input.value = String(defaults[input.dataset.ray]); });
    render();
  });

  render();
})();
