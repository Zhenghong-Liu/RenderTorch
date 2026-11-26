(() => {
  const mount = document.getElementById("pixel-demo");
  if (!mount) return;

  const width = 8;
  const height = 6;
  const selected = { x: 5, y: 2 };
  const pixels = new Uint8ClampedArray(width * height * 4);
  function resetPixels() {
    selected.x = 5;
    selected.y = 2;
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const gray = Math.round((255 * x) / (width - 1));
        const offset = (y * width + x) * 4;
        pixels.set([gray, gray, gray, 255], offset);
      }
    }
    pixels.set([12, 200, 64, 255], (selected.y * width + selected.x) * 4);
  }

  mount.insertAdjacentHTML(
    "beforeend",
    `<div class="demo-toolbar">
      <label>x <input type="number" data-x min="0" max="7" step="1" value="5"></label>
      <label>y <input type="number" data-y min="0" max="5" step="1" value="2"></label>
      <label>R <input type="range" data-channel="0" min="0" max="255" value="12"><output>12</output></label>
      <label>G <input type="range" data-channel="1" min="0" max="255" value="200"><output>200</output></label>
      <label>B <input type="range" data-channel="2" min="0" max="255" value="64"><output>64</output></label>
      <label>A <input type="range" data-channel="3" min="0" max="255" value="255"><output>255</output></label>
      <button type="button" data-reset>重置</button>
    </div>
    <canvas id="pixel-canvas" width="640" height="360" role="img" aria-label="八列六行像素网格，点击格子选择像素"></canvas>
    <p class="demo-readout" data-readout aria-live="polite"></p>`
  );

  const canvas = mount.querySelector("#pixel-canvas");
  const context = canvas.getContext("2d");
  const xInput = mount.querySelector("[data-x]");
  const yInput = mount.querySelector("[data-y]");
  const readout = mount.querySelector("[data-readout]");
  const channels = [...mount.querySelectorAll("[data-channel]")];

  function currentOffset() {
    return (selected.y * width + selected.x) * 4;
  }

  function syncChannels() {
    const offset = currentOffset();
    channels.forEach((input) => {
      const value = pixels[offset + Number(input.dataset.channel)];
      input.value = String(value);
      input.nextElementSibling.value = String(value);
    });
  }

  function draw() {
    const cellWidth = canvas.width / width;
    const cellHeight = canvas.height / height;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#10242c";
    context.fillRect(0, 0, canvas.width, canvas.height);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const offset = (y * width + x) * 4;
        const alpha = pixels[offset + 3] / 255;
        context.fillStyle = `rgba(${pixels[offset]}, ${pixels[offset + 1]}, ${pixels[offset + 2]}, ${alpha})`;
        context.fillRect(x * cellWidth + 1, y * cellHeight + 1, cellWidth - 2, cellHeight - 2);
        context.strokeStyle = "rgba(255,255,255,.16)";
        context.strokeRect(x * cellWidth + 0.5, y * cellHeight + 0.5, cellWidth - 1, cellHeight - 1);
        if (x === selected.x && y === selected.y) {
          context.strokeStyle = "#ffd48a";
          context.lineWidth = 4;
          context.strokeRect(x * cellWidth + 3, y * cellHeight + 3, cellWidth - 6, cellHeight - 6);
          context.lineWidth = 1;
        }
      }
    }
    const index = selected.y * width + selected.x;
    const offset = currentOffset();
    readout.textContent =
      `像素 (${selected.x}, ${selected.y}) · index=${index} · ` +
      `u=${window.RTBook.formatNumber(selected.x / (width - 1), 4)} · ` +
      `v=${window.RTBook.formatNumber(selected.y / (height - 1), 4)} · ` +
      `byte_offset=${offset} · RGBA=(${pixels[offset]}, ${pixels[offset + 1]}, ${pixels[offset + 2]}, ${pixels[offset + 3]})`;
  }

  function selectPixel(x, y) {
    selected.x = window.RTBook.clamp(Math.round(x), 0, width - 1);
    selected.y = window.RTBook.clamp(Math.round(y), 0, height - 1);
    xInput.value = String(selected.x);
    yInput.value = String(selected.y);
    syncChannels();
    draw();
  }

  xInput.addEventListener("input", () => selectPixel(Number(xInput.value), selected.y));
  yInput.addEventListener("input", () => selectPixel(selected.x, Number(yInput.value)));
  channels.forEach((input) => {
    input.addEventListener("input", () => {
      pixels[currentOffset() + Number(input.dataset.channel)] = Number(input.value);
      input.nextElementSibling.value = input.value;
      draw();
    });
  });
  canvas.addEventListener("click", (event) => {
    const bounds = canvas.getBoundingClientRect();
    const x = ((event.clientX - bounds.left) / bounds.width) * width;
    const y = ((event.clientY - bounds.top) / bounds.height) * height;
    selectPixel(x, y);
  });

  mount.querySelector("[data-reset]").addEventListener("click", () => {
    resetPixels();
    xInput.value = String(selected.x);
    yInput.value = String(selected.y);
    channels.forEach((input, index) => {
      input.value = String([12, 200, 64, 255][index]);
    });
    syncChannels();
    draw();
  });

  resetPixels();
  draw();
})();
