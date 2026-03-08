/*
README (quick start)
1) Run locally with any static server, for example:
   - Python: python3 -m http.server 8080
   - Node: npx serve .
2) Open http://localhost:8080 on mobile (or desktop for testing).
3) Allow camera access (or upload a photo), then tap "Scan Dominoes".

Known limitations
- Heavy overlap can merge two dominoes into one contour.
- Very dark scenes or strong glare can still hide some pips.
- Busy patterned tables can still produce occasional false positives.

Ideas to improve accuracy
- Add perspective warp from 4 corners before pip analysis per tile.
- Add temporal smoothing over several frames for stable totals.
- Add a tiny on-device ML detector for domino proposals + confidence fusion.
*/

(() => {
  const elements = {
    video: document.getElementById("camera"),
    overlayCanvas: document.getElementById("overlayCanvas"),
    captureCanvas: document.getElementById("captureCanvas"),
    debugCanvas: document.getElementById("debugCanvas"),
    scanBtn: document.getElementById("scanBtn"),
    resetBtn: document.getElementById("resetBtn"),
    uploadInput: document.getElementById("uploadInput"),
    totalPips: document.getElementById("totalPips"),
    dominoCount: document.getElementById("dominoCount"),
    perDomino: document.getElementById("perDomino"),
    status: document.getElementById("status"),
    debugToggle: document.getElementById("debugToggle"),
    debugControls: document.getElementById("debugControls"),
    debugStage: document.getElementById("debugStage")
  };

  const state = {
    opencvReady: false,
    cameraReady: false,
    stream: null,
    sourceMode: "camera",
    uploadedImageBitmap: null,
    debugEnabled: false,
    latestDebugStage: "combined",
    debugMats: new Map()
  };

  function setStatus(text, isError = false) {
    elements.status.textContent = text;
    elements.status.classList.toggle("error", isError);
  }

  function waitForOpenCv() {
    return new Promise((resolve, reject) => {
      const timeoutMs = 15000;
      const start = performance.now();
      const check = () => {
        if (window.cv && window.cv.Mat) {
          resolve();
          return;
        }
        if (performance.now() - start > timeoutMs) {
          reject(new Error("OpenCV.js failed to load in time"));
          return;
        }
        setTimeout(check, 100);
      };
      check();
    });
  }

  function syncCanvasSizes(width, height) {
    const w = width || elements.video.videoWidth;
    const h = height || elements.video.videoHeight;
    if (!w || !h) {
      return;
    }
    [elements.overlayCanvas, elements.captureCanvas, elements.debugCanvas].forEach((canvas) => {
      canvas.width = w;
      canvas.height = h;
    });
  }

  function clearDebugMats() {
    state.debugMats.forEach((mat) => mat.delete());
    state.debugMats.clear();
  }

  function storeDebugMat(name, mat) {
    if (!state.debugEnabled) {
      return;
    }
    if (state.debugMats.has(name)) {
      state.debugMats.get(name).delete();
    }
    state.debugMats.set(name, mat.clone());
  }

  function renderDebugStage(stage) {
    if (!state.debugEnabled) {
      return;
    }
    const mat = state.debugMats.get(stage);
    if (mat) {
      cv.imshow(elements.debugCanvas, mat);
    }
  }

  function drawRotatedRect(ctx, points, color) {
    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i += 1) {
      ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.closePath();
    ctx.stroke();
  }

  function drawOverlays(detections) {
    const ctx = elements.overlayCanvas.getContext("2d");
    ctx.clearRect(0, 0, elements.overlayCanvas.width, elements.overlayCanvas.height);

    ctx.lineWidth = Math.max(2, Math.round(elements.overlayCanvas.width / 240));
    ctx.font = `${Math.max(12, Math.round(elements.overlayCanvas.width / 40))}px sans-serif`;

    detections.forEach((detection, idx) => {
      const color = detection.confidence >= 0.7 ? "#5dffa1" : "#ffd166";
      drawRotatedRect(ctx, detection.points, color);

      const anchor = detection.points.reduce((top, point) => (point.y < top.y ? point : top), detection.points[0]);
      const label = `#${idx + 1} ${detection.pips}p (${Math.round(detection.confidence * 100)}%)`;
      const tw = ctx.measureText(label).width + 10;
      const y = Math.max(18, anchor.y - 6);
      ctx.fillStyle = "rgba(0,0,0,0.75)";
      ctx.fillRect(anchor.x, y - 16, tw, 18);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, anchor.x + 5, y - 3);
    });
  }

  function clearUiResults() {
    elements.totalPips.textContent = "0";
    elements.dominoCount.textContent = "0";
    elements.perDomino.textContent = "-";
    const overlayCtx = elements.overlayCanvas.getContext("2d");
    overlayCtx.clearRect(0, 0, elements.overlayCanvas.width, elements.overlayCanvas.height);
    const debugCtx = elements.debugCanvas.getContext("2d");
    debugCtx.clearRect(0, 0, elements.debugCanvas.width, elements.debugCanvas.height);
    clearDebugMats();
  }

  function getFrameFromSource() {
    const ctx = elements.captureCanvas.getContext("2d", { willReadFrequently: true });

    if (state.sourceMode === "upload" && state.uploadedImageBitmap) {
      syncCanvasSizes(state.uploadedImageBitmap.width, state.uploadedImageBitmap.height);
      ctx.drawImage(state.uploadedImageBitmap, 0, 0);
      return true;
    }

    if (!state.cameraReady || !elements.video.videoWidth || !elements.video.videoHeight) {
      return false;
    }

    syncCanvasSizes();
    ctx.drawImage(elements.video, 0, 0, elements.captureCanvas.width, elements.captureCanvas.height);
    return true;
  }

  function computeIoU(a, b) {
    const ax2 = a.x + a.w;
    const ay2 = a.y + a.h;
    const bx2 = b.x + b.w;
    const by2 = b.y + b.h;
    const ix = Math.max(0, Math.min(ax2, bx2) - Math.max(a.x, b.x));
    const iy = Math.max(0, Math.min(ay2, by2) - Math.max(a.y, b.y));
    const intersection = ix * iy;
    const union = a.w * a.h + b.w * b.h - intersection;
    return union > 0 ? intersection / union : 0;
  }

  function nonMaxSuppression(detections) {
    const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
    const kept = [];
    for (const candidate of sorted) {
      const overlaps = kept.some((selected) => computeIoU(candidate.rect, selected.rect) > 0.35);
      if (!overlaps) {
        kept.push(candidate);
      }
    }
    return kept;
  }

  function rotatedPointsFromRect(rotRect) {
    const box = new cv.Mat();
    cv.boxPoints(rotRect, box);
    const points = [];
    for (let i = 0; i < 4; i += 1) {
      points.push({ x: box.data32F[i * 2], y: box.data32F[i * 2 + 1] });
    }
    box.delete();
    return points;
  }

  function clamp01(value) {
    return Math.max(0, Math.min(1, value));
  }

  function countPipsInRect(gray, rect) {
    const x = Math.max(0, rect.x);
    const y = Math.max(0, rect.y);
    const width = Math.min(gray.cols - x, rect.width);
    const height = Math.min(gray.rows - y, rect.height);

    if (width < 20 || height < 20) {
      return { pipCount: 0, pipConfidence: 0, pipMask: null, meanIntensity: 0 };
    }

    const roi = gray.roi(new cv.Rect(x, y, width, height));
    const roiBlur = new cv.Mat();
    const pipMask = new cv.Mat();
    const pipContours = new cv.MatVector();
    const pipHierarchy = new cv.Mat();

    const meanIntensity = cv.mean(roi)[0];

    cv.GaussianBlur(roi, roiBlur, new cv.Size(5, 5), 0);
    cv.threshold(roiBlur, pipMask, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);

    const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3));
    cv.morphologyEx(pipMask, pipMask, cv.MORPH_OPEN, kernel);
    kernel.delete();

    cv.findContours(pipMask, pipContours, pipHierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    let pipCount = 0;
    const roiArea = width * height;
    let circularHits = 0;

    for (let i = 0; i < pipContours.size(); i += 1) {
      const blob = pipContours.get(i);
      const area = cv.contourArea(blob);
      if (area < roiArea * 0.0008 || area > roiArea * 0.035) {
        blob.delete();
        continue;
      }

      const perimeter = cv.arcLength(blob, true);
      if (!perimeter) {
        blob.delete();
        continue;
      }

      const circularity = (4 * Math.PI * area) / (perimeter * perimeter);
      const bRect = cv.boundingRect(blob);
      const ar = bRect.width / Math.max(1, bRect.height);
      const centerX = bRect.x + bRect.width / 2;
      const centerY = bRect.y + bRect.height / 2;
      const edgePaddingX = width * 0.06;
      const edgePaddingY = height * 0.06;
      const insideTile =
        centerX > edgePaddingX && centerX < width - edgePaddingX && centerY > edgePaddingY && centerY < height - edgePaddingY;

      if (insideTile && circularity > 0.32 && ar > 0.45 && ar < 1.9) {
        pipCount += 1;
        circularHits += circularity;
      }
      blob.delete();
    }

    const boundedCount = Math.min(12, pipCount);
    const avgCircularity = pipCount > 0 ? circularHits / pipCount : 0;
    const countScore = boundedCount > 0 ? clamp01(1 - Math.abs(6 - boundedCount) / 7) : 0;
    const pipConfidence = clamp01(avgCircularity * 0.7 + countScore * 0.3);

    roi.delete();
    roiBlur.delete();
    pipContours.delete();
    pipHierarchy.delete();

    return { pipCount: boundedCount, pipConfidence, pipMask, meanIntensity };
  }

  function analyzeCurrentFrame() {
    if (!state.opencvReady) {
      setStatus("OpenCV loading. Please wait...", true);
      return;
    }

    if (!getFrameFromSource()) {
      setStatus("No frame available. Allow camera or upload a photo.", true);
      return;
    }

    setStatus("Scanning...");
    clearDebugMats();

    let src;
    let gray;
    let normalized;
    let threshold;
    let edges;
    let combined;
    let contours;
    let hierarchy;

    try {
      src = cv.imread(elements.captureCanvas);
      gray = new cv.Mat();
      normalized = new cv.Mat();
      threshold = new cv.Mat();
      edges = new cv.Mat();
      combined = new cv.Mat();
      contours = new cv.MatVector();
      hierarchy = new cv.Mat();

      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
      cv.equalizeHist(gray, normalized);

      const blur = new cv.Mat();
      const brightMask = new cv.Mat();
      cv.GaussianBlur(normalized, blur, new cv.Size(5, 5), 0);

      cv.adaptiveThreshold(
        blur,
        threshold,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        35,
        -4
      );
      cv.threshold(blur, brightMask, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
      cv.bitwise_or(threshold, brightMask, threshold);

      cv.Canny(blur, edges, 45, 120);
      const edgeKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
      cv.dilate(edges, edges, edgeKernel);
      cv.bitwise_or(threshold, edges, combined);
      cv.morphologyEx(combined, combined, cv.MORPH_CLOSE, edgeKernel);
      cv.morphologyEx(combined, combined, cv.MORPH_OPEN, edgeKernel);
      edgeKernel.delete();
      blur.delete();
      brightMask.delete();

      storeDebugMat("normalized", normalized);
      storeDebugMat("threshold", threshold);
      storeDebugMat("edges", edges);
      storeDebugMat("combined", combined);

      cv.findContours(combined, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

      const frameArea = src.cols * src.rows;
      const detections = [];
      let lastPipMask = null;

      for (let i = 0; i < contours.size(); i += 1) {
        const contour = contours.get(i);
        const area = cv.contourArea(contour);
        if (area < frameArea * 0.0025 || area > frameArea * 0.75) {
          contour.delete();
          continue;
        }

        const perimeter = cv.arcLength(contour, true);
        if (!perimeter) {
          contour.delete();
          continue;
        }

        const approx = new cv.Mat();
        cv.approxPolyDP(contour, approx, 0.03 * perimeter, true);

        const rotRect = cv.minAreaRect(contour);
        const width = Math.max(rotRect.size.width, rotRect.size.height);
        const height = Math.max(1, Math.min(rotRect.size.width, rotRect.size.height));
        const ratio = width / height;

        const rectArea = rotRect.size.width * rotRect.size.height;
        const rectangularity = rectArea > 0 ? area / rectArea : 0;

        const hull = new cv.Mat();
        cv.convexHull(contour, hull, false, true);
        const hullArea = cv.contourArea(hull);
        const solidity = hullArea > 0 ? area / hullArea : 0;

        const bbox = cv.boundingRect(contour);

        const angleScore = ratio >= 1.1 && ratio <= 3.8 ? 1 : 0;
        const shapeScore = clamp01((rectangularity - 0.3) / 0.55);
        const solidityScore = clamp01((solidity - 0.45) / 0.5);

        if (ratio < 1.02 || ratio > 4.2 || rectangularity < 0.26 || solidity < 0.42 || approx.rows < 4 || approx.rows > 14) {
          approx.delete();
          hull.delete();
          contour.delete();
          continue;
        }

        const { pipCount, pipConfidence, pipMask, meanIntensity } = countPipsInRect(gray, bbox);
        if (lastPipMask) {
          lastPipMask.delete();
        }
        lastPipMask = pipMask;

        const lightScore = clamp01((meanIntensity - 85) / 110);
        const geometryConfidence = 0.5 * shapeScore + 0.25 * solidityScore + 0.15 * angleScore + 0.1 * lightScore;
        const pipScore = Math.min(1, pipConfidence + (pipCount > 0 ? 0.2 : 0));
        const confidence = 0.62 * geometryConfidence + 0.38 * pipScore;

        const points = rotatedPointsFromRect(rotRect);

        detections.push({
          rect: { x: bbox.x, y: bbox.y, w: bbox.width, h: bbox.height },
          points,
          pips: pipCount,
          confidence,
          meta: {
            ratio: Number(ratio.toFixed(2)),
            rectangularity: Number(rectangularity.toFixed(2)),
            solidity: Number(solidity.toFixed(2)),
            pipConfidence: Number(pipConfidence.toFixed(2)),
            geometryConfidence: Number(geometryConfidence.toFixed(2)),
            meanIntensity: Number(meanIntensity.toFixed(1))
          }
        });

        console.debug("dominoCandidate", {
          i,
          area: Math.round(area),
          ratio: ratio.toFixed(2),
          rectangularity: rectangularity.toFixed(2),
          solidity: solidity.toFixed(2),
          pips: pipCount,
          confidence: confidence.toFixed(2)
        });

        approx.delete();
        hull.delete();
        contour.delete();
      }

      if (lastPipMask) {
        storeDebugMat("pip", lastPipMask);
        lastPipMask.delete();
      }

      const nmsDetections = nonMaxSuppression(detections);
      let filtered = nmsDetections.filter((d) => d.confidence >= 0.22);
      const usedRelaxedFallback = filtered.length === 0 && nmsDetections.length > 0;
      if (usedRelaxedFallback) {
        filtered = nmsDetections.filter((d) => d.meta.geometryConfidence >= 0.2);
      }
      drawOverlays(filtered);

      const total = filtered.reduce((sum, domino) => sum + domino.pips, 0);
      elements.totalPips.textContent = String(total);
      elements.dominoCount.textContent = String(filtered.length);
      elements.perDomino.textContent = filtered.length
        ? filtered.map((d, idx) => `#${idx + 1}: ${d.pips} (${Math.round(d.confidence * 100)}%)`).join(" · ")
        : "-";

      if (!filtered.length) {
        setStatus("Detection uncertain, please rescan", true);
      } else {
        const uncertain = filtered.filter((d) => d.confidence < 0.6 || d.pips === 0).length;
        if (usedRelaxedFallback) {
          setStatus(`${filtered.length} possible dominoes detected (low confidence)`, true);
        } else if (uncertain > 0) {
          setStatus(`${filtered.length} dominoes detected (${uncertain} low-confidence)`);
        } else {
          setStatus(`${filtered.length} dominoes detected`);
        }
      }

      state.latestDebugStage = elements.debugStage.value;
      renderDebugStage(state.latestDebugStage);

      console.info("scanSummary", {
        sourceMode: state.sourceMode,
        contourCount: contours.size(),
        dominoes: filtered.length,
        totalPips: total,
        perDomino: filtered.map((d) => ({ pips: d.pips, confidence: d.confidence, meta: d.meta }))
      });
    } catch (error) {
      console.error("scanFailed", error);
      setStatus(`Scan failed: ${error.message}`, true);
    } finally {
      [src, gray, normalized, threshold, edges, combined, hierarchy].forEach((mat) => mat && mat.delete());
      if (contours) {
        contours.delete();
      }
    }
  }

  async function initCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      });

      state.stream = stream;
      elements.video.srcObject = stream;

      await new Promise((resolve) => {
        elements.video.onloadedmetadata = () => resolve();
      });

      state.cameraReady = true;
      state.sourceMode = "camera";
      syncCanvasSizes();
      setStatus("Camera ready");
      elements.scanBtn.disabled = !state.opencvReady;
    } catch (error) {
      state.cameraReady = false;
      elements.scanBtn.disabled = false;
      setStatus("Camera unavailable. Please upload a photo.", true);
      console.warn("cameraInitFailed", error);
    }
  }

  async function init() {
    setStatus("Loading OpenCV...");

    try {
      await waitForOpenCv();
      state.opencvReady = true;
      setStatus("OpenCV ready. Initializing camera...");
    } catch (error) {
      setStatus(error.message, true);
      console.error(error);
      return;
    }

    await initCamera();
  }

  function bindEvents() {
    elements.scanBtn.addEventListener("click", analyzeCurrentFrame);

    elements.resetBtn.addEventListener("click", () => {
      clearUiResults();
      state.sourceMode = state.cameraReady ? "camera" : "upload";
      setStatus(state.cameraReady ? "Ready to scan" : "Upload a photo to scan");
    });

    elements.uploadInput.addEventListener("change", async (event) => {
      const file = event.target.files?.[0];
      if (!file) {
        return;
      }
      try {
        const imageBitmap = await createImageBitmap(file);
        state.uploadedImageBitmap = imageBitmap;
        state.sourceMode = "upload";
        syncCanvasSizes(imageBitmap.width, imageBitmap.height);

        const previewCtx = elements.captureCanvas.getContext("2d");
        previewCtx.drawImage(imageBitmap, 0, 0);
        setStatus("Photo ready. Tap Scan Dominoes.");
        elements.scanBtn.disabled = !state.opencvReady;
      } catch (error) {
        setStatus(`Could not read photo: ${error.message}`, true);
      }
    });

    elements.debugToggle.addEventListener("change", () => {
      state.debugEnabled = elements.debugToggle.checked;
      elements.debugControls.hidden = !state.debugEnabled;
      elements.debugCanvas.hidden = !state.debugEnabled;
      if (!state.debugEnabled) {
        const ctx = elements.debugCanvas.getContext("2d");
        ctx.clearRect(0, 0, elements.debugCanvas.width, elements.debugCanvas.height);
      } else {
        renderDebugStage(elements.debugStage.value);
      }
    });

    elements.debugStage.addEventListener("change", () => {
      state.latestDebugStage = elements.debugStage.value;
      renderDebugStage(state.latestDebugStage);
    });

    window.addEventListener("resize", () => {
      if (state.sourceMode === "camera" && state.cameraReady) {
        syncCanvasSizes();
      }
    });

    window.addEventListener("beforeunload", () => {
      if (state.stream) {
        state.stream.getTracks().forEach((track) => track.stop());
      }
      clearDebugMats();
    });
  }

  bindEvents();
  init();
})();
