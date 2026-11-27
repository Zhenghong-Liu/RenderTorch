(() => {
  const mount = document.getElementById("lighting-demo");
  if (!mount) return;

  mount.insertAdjacentHTML(
    "beforeend",
    `<div class="demo-toolbar">
      <label>光源方位角 <input type="range" data-azimuth min="-180" max="180" step="1" value="37"><output>37°</output></label>
      <label>光源仰角 <input type="range" data-elevation min="-90" max="90" step="1" value="0"><output>0°</output></label>
      <label>环境光强度 <input type="range" data-ambient min="0" max="1" step="0.01" value="0.1"><output>0.10</output></label>
      <label>高光指数 <input type="range" data-shininess min="1" max="128" step="1" value="16"><output>16</output></label>
      <button type="button" data-reset>重置</button>
    </div>
    <canvas id="lighting-canvas" width="721" height="421" role="img" aria-label="逐像素计算 Lambert 漫反射与 Blinn-Phong 高光的球体圆盘" aria-describedby="lighting-readout">逐像素着色的球体圆盘</canvas>
    <p class="demo-readout" id="lighting-readout" data-readout aria-live="polite"></p>`
  );

  const canvas = mount.querySelector("#lighting-canvas");
  const context = canvas.getContext("2d");
  if (!context) return;
  const readout = mount.querySelector("[data-readout]");
  const inputs = [...mount.querySelectorAll("input[type='range']")];
  const reset = mount.querySelector("[data-reset]");
  const defaults = { azimuth: "37", elevation: "0", ambient: "0.1", shininess: "16" };
  const baseColor = [0.6, 0.2, 0.1];
  const diffuseStrength = 0.8;
  const specularStrength = 0.5;
  const specularColor = [1, 1, 1];
  const centerX = Math.floor(canvas.width / 2);
  const centerY = Math.floor(canvas.height / 2);
  const radius = Math.min(canvas.width, canvas.height) * 0.38;
  const image = context.createImageData(canvas.width, canvas.height);
  const viewer = [0, 0, 1];
  const clamp = (value) => Math.min(1, Math.max(0, value));
  const inputKey = (input) => Object.keys(input.dataset)[0];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const fmt = (value) => (Math.abs(value) < 0.0005 ? 0 : value).toFixed(3);
  const fmtPoint = (point) => `(${point.map(fmt).join(", ")})`;
  const fmtRgb = (values) => `(${values.join(", ")})`;

  function normalize(vector) {
    const length = Math.hypot(...vector);
    return length > 1e-12 ? vector.map((value) => value / length) : [0, 0, 0];
  }

  function shade(normal, light, half, ambientStrength, shininess) {
    const diffuse = Math.max(0, dot(normal, light));
    const specularFactor = Math.pow(Math.max(0, dot(normal, half)), shininess);
    const channels = baseColor.map((base, channel) => {
      const ambient = base * ambientStrength;
      const diffuseContribution = diffuseStrength * base * diffuse;
      const specularContribution = specularStrength * specularColor[channel] * specularFactor;
      const byte = Math.round(clamp(ambient + diffuseContribution + specularContribution) * 255);
      const ambientByte = Math.floor(clamp(ambient) * 255);
      const diffuseByte = Math.floor(clamp(ambient + diffuseContribution) * 255) - ambientByte;
      return {
        byte,
        ambientByte,
        diffuseByte,
        specularByte: byte - ambientByte - diffuseByte,
      };
    });
    return { diffuse, specular: specularFactor, channels };
  }

  function update() {
    const parameters = Object.fromEntries(inputs.map((input) => [inputKey(input), Number(input.value)]));
    const azimuth = parameters.azimuth * Math.PI / 180;
    const elevation = parameters.elevation * Math.PI / 180;
    const light = normalize([
      Math.cos(elevation) * Math.sin(azimuth),
      Math.sin(elevation),
      Math.cos(elevation) * Math.cos(azimuth),
    ]);
    const half = normalize([light[0] + viewer[0], light[1] + viewer[1], light[2] + viewer[2]]);
    const data = image.data;

    for (let y = 0; y < canvas.height; y += 1) {
      for (let x = 0; x < canvas.width; x += 1) {
        const offset = (y * canvas.width + x) * 4;
        const nx = (x - centerX) / radius;
        const ny = -(y - centerY) / radius;
        const distanceSquared = nx * nx + ny * ny;
        if (distanceSquared > 1) {
          data[offset] = 16;
          data[offset + 1] = 36;
          data[offset + 2] = 44;
        } else {
          const normal = normalize([nx, ny, Math.sqrt(Math.max(0, 1 - distanceSquared))]);
          const result = shade(normal, light, half, parameters.ambient, parameters.shininess);
          result.channels.forEach((channel, index) => { data[offset + index] = channel.byte; });
        }
        data[offset + 3] = 255;
      }
    }

    context.putImageData(image, 0, 0);
    context.strokeStyle = "rgba(246, 243, 235, 0.28)";
    context.lineWidth = 1;
    context.beginPath();
    context.arc(centerX, centerY, radius, 0, Math.PI * 2);
    context.stroke();
    context.strokeStyle = "#ffd48a";
    context.beginPath();
    context.arc(centerX, centerY, 7, 0, Math.PI * 2);
    context.stroke();
    context.fillStyle = "#9db6b8";
    context.font = "14px system-ui";
    context.fillText("圆盘中心为读数样本", 22, 28);

    inputs.forEach((input) => {
      const key = inputKey(input);
      const value = Number(input.value);
      input.nextElementSibling.value = key === "azimuth" || key === "elevation" ? `${value}°` : key === "ambient" ? value.toFixed(2) : String(value);
    });

    const sampleNormal = [0, 0, 1];
    const sample = shade(sampleNormal, light, half, parameters.ambient, parameters.shininess);
    const channelValue = (key) => sample.channels.map((channel) => channel[key]);
    readout.textContent =
      `圆盘中心样本：N=${fmtPoint(sampleNormal)} · L=${fmtPoint(light)} · V=${fmtPoint(viewer)} · H=${fmtPoint(half)} · ` +
      `环境光强度=${fmt(parameters.ambient)} · Lambert 漫反射=${fmt(sample.diffuse)} · Blinn-Phong 高光=${fmt(sample.specular)}（指数=${parameters.shininess}） · ` +
      `8位分量：环境光 RGB=${fmtRgb(channelValue("ambientByte"))} + 漫反射 RGB=${fmtRgb(channelValue("diffuseByte"))} + ` +
      `高光 RGB=${fmtRgb(channelValue("specularByte"))} = 最终 RGB=${fmtRgb(channelValue("byte"))}`;
  }

  inputs.forEach((input) => input.addEventListener("input", update));
  reset.addEventListener("click", () => {
    inputs.forEach((input) => {
      input.value = defaults[inputKey(input)];
    });
    update();
  });

  update();
})();
