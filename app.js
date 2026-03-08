(() => {
  const { useEffect, useRef, useState } = React;

  /**
   * Poll until OpenCV runtime is available.
   * @returns {Promise<void>}
   */
  function waitForOpenCv() {
    return new Promise((resolve) => {
      const check = () => {
        if (window.cv && window.cv.Mat) {
          resolve();
        } else {
          setTimeout(check, 120);
        }
      };
      check();
    });
  }

  function App() {
    const videoRef = useRef(null);
    const overlayRef = useRef(null);
    const captureRef = useRef(null);

    const [opencvReady, setOpenCvReady] = useState(false);
    const [cameraReady, setCameraReady] = useState(false);
    const [status, setStatus] = useState({ text: "Initializing camera...", isError: false });
    const [pipTotal, setPipTotal] = useState(0);
    const [dominoCount, setDominoCount] = useState(0);

    const streamRef = useRef(null);

    /**
     * Update status text with optional error style.
     */
    const updateStatus = (text, isError = false) => setStatus({ text, isError });

    /**
     * Keep overlay and capture canvas in sync with camera frame dimensions.
     */
    const syncCanvasSizes = () => {
      const video = videoRef.current;
      const overlay = overlayRef.current;
      const capture = captureRef.current;
      if (!video || !overlay || !capture || !video.videoWidth || !video.videoHeight) return;

      overlay.width = video.videoWidth;
      overlay.height = video.videoHeight;
      capture.width = video.videoWidth;
      capture.height = video.videoHeight;
    };

    /**
     * Draw box + label for each detected domino tile.
     * @param {Array<{rect: {x:number,y:number,w:number,h:number}, pips:number}>} detections
     */
    const drawDetections = (detections) => {
      const overlay = overlayRef.current;
      if (!overlay) return;

      const ctx = overlay.getContext("2d");
      ctx.clearRect(0, 0, overlay.width, overlay.height);
      ctx.lineWidth = 4;
      ctx.font = "20px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";

      detections.forEach((detection, idx) => {
        const { x, y, w, h } = detection.rect;
        ctx.strokeStyle = "#00ffa3";
        ctx.strokeRect(x, y, w, h);

        ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
        ctx.fillRect(x, Math.max(0, y - 30), 170, 28);

        ctx.fillStyle = "#fff";
        ctx.fillText(`#${idx + 1} pips: ${detection.pips}`, x + 7, Math.max(20, y - 10));
      });
    };

    /**
     * Run a heuristic CV pipeline:
     * 1) contour domino-like rectangles
     * 2) count circular pip features per rectangle with HoughCircles.
     */
    const processFrame = () => {
      const video = videoRef.current;
      const capture = captureRef.current;
      const overlay = overlayRef.current;
      if (!video || !capture || !overlay) return;

      const captureCtx = capture.getContext("2d");
      captureCtx.drawImage(video, 0, 0, capture.width, capture.height);

      const src = cv.imread(capture);
      const gray = new cv.Mat();
      const blur = new cv.Mat();
      const thresh = new cv.Mat();

      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
      cv.GaussianBlur(gray, blur, new cv.Size(5, 5), 0);
      cv.adaptiveThreshold(
        blur,
        thresh,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,
        41,
        7
      );

      const contours = new cv.MatVector();
      const hierarchy = new cv.Mat();
      cv.findContours(thresh, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

      const detections = [];

      for (let i = 0; i < contours.size(); i += 1) {
        const contour = contours.get(i);
        const area = cv.contourArea(contour);

        if (area < 2200 || area > src.cols * src.rows * 0.35) {
          contour.delete();
          continue;
        }

        const rect = cv.boundingRect(contour);
        const ratio = Math.max(rect.width, rect.height) / Math.min(rect.width, rect.height);
        if (ratio < 1.35 || ratio > 2.75) {
          contour.delete();
          continue;
        }

        const roi = gray.roi(new cv.Rect(rect.x, rect.y, rect.width, rect.height));
        const circles = new cv.Mat();

        cv.HoughCircles(
          roi,
          circles,
          cv.HOUGH_GRADIENT,
          1,
          Math.max(8, Math.min(rect.width, rect.height) / 8),
          85,
          13,
          4,
          Math.floor(Math.min(rect.width, rect.height) / 4)
        );

        detections.push({
          rect: { x: rect.x, y: rect.y, w: rect.width, h: rect.height },
          pips: circles.cols
        });

        circles.delete();
        roi.delete();
        contour.delete();
      }

      const total = detections.reduce((sum, item) => sum + item.pips, 0);
      setPipTotal(total);
      setDominoCount(detections.length);
      drawDetections(detections);

      if (detections.length > 0) {
        updateStatus(`Scan complete: ${detections.length} domino(es), ${total} total pips.`);
      } else {
        updateStatus("No dominoes detected. Try brighter light and a higher-contrast table.");
      }

      src.delete();
      gray.delete();
      blur.delete();
      thresh.delete();
      contours.delete();
      hierarchy.delete();
    };

    const handleScan = () => {
      if (!opencvReady) {
        updateStatus("OpenCV is still loading. Please wait...", true);
        return;
      }
      if (!cameraReady) {
        updateStatus("Camera not ready yet.", true);
        return;
      }
      syncCanvasSizes();
      processFrame();
    };

    const handleReset = () => {
      const overlay = overlayRef.current;
      if (overlay) {
        const ctx = overlay.getContext("2d");
        ctx.clearRect(0, 0, overlay.width, overlay.height);
      }
      setPipTotal(0);
      setDominoCount(0);
      updateStatus("Reset complete. Ready for next scan.");
    };

    useEffect(() => {
      let alive = true;

      const init = async () => {
        try {
          // Load OpenCV runtime first so scan is available quickly after camera warmup.
          await waitForOpenCv();
          if (!alive) return;
          setOpenCvReady(true);

          const stream = await navigator.mediaDevices.getUserMedia({
            video: {
              facingMode: { ideal: "environment" },
              width: { ideal: 1280 },
              height: { ideal: 720 }
            },
            audio: false
          });

          if (!alive) {
            stream.getTracks().forEach((t) => t.stop());
            return;
          }

          streamRef.current = stream;
          const video = videoRef.current;
          video.srcObject = stream;

          await new Promise((resolve) => {
            video.onloadedmetadata = resolve;
          });

          syncCanvasSizes();
          setCameraReady(true);
          updateStatus("Camera ready. Place dominoes flat and tap Scan Dominoes.");
        } catch (error) {
          updateStatus(`Initialization failed: ${error.message}`, true);
        }
      };

      init();
      window.addEventListener("resize", syncCanvasSizes);
      window.addEventListener("orientationchange", syncCanvasSizes);

      return () => {
        alive = false;
        window.removeEventListener("resize", syncCanvasSizes);
        window.removeEventListener("orientationchange", syncCanvasSizes);
        if (streamRef.current) {
          streamRef.current.getTracks().forEach((track) => track.stop());
        }
      };
    }, []);

    return React.createElement(
      "main",
      { className: "app-shell" },
      React.createElement(
        "section",
        { className: "col-main" },
        React.createElement(
          "header",
          { className: "card" },
          React.createElement("h1", null, "Countino"),
          React.createElement(
            "p",
            { className: "subtitle" },
            "React mobile app for automatic domino pip counting on iPhone and Android."
          )
        ),
        React.createElement(
          "section",
          { className: "card" },
          React.createElement(
            "div",
            { className: "camera-frame" },
            React.createElement("video", {
              ref: videoRef,
              className: "camera-layer",
              autoPlay: true,
              playsInline: true,
              muted: true
            }),
            React.createElement("canvas", { ref: overlayRef, className: "camera-layer overlay-layer" })
          )
        ),
        React.createElement(
          "section",
          { className: "controls" },
          React.createElement(
            "button",
            {
              type: "button",
              className: "primary",
              onClick: handleScan,
              disabled: !opencvReady || !cameraReady
            },
            "Scan Dominoes"
          ),
          React.createElement(
            "button",
            {
              type: "button",
              className: "secondary",
              onClick: handleReset
            },
            "Reset"
          )
        )
      ),
      React.createElement(
        "section",
        { className: "col-side" },
        React.createElement(
          "section",
          { className: "card", "aria-live": "polite" },
          React.createElement("h2", null, "Scan Results"),
          React.createElement(
            "p",
            { className: "metric" },
            React.createElement("strong", null, "Total pip count: "),
            String(pipTotal)
          ),
          React.createElement(
            "p",
            { className: "metric" },
            React.createElement("strong", null, "Dominoes detected: "),
            String(dominoCount)
          ),
          React.createElement("p", { className: `status${status.isError ? " error" : ""}` }, status.text)
        ),
        React.createElement(
          "section",
          { className: "card" },
          React.createElement("h2", null, "How to use"),
          React.createElement(
            "ol",
            null,
            React.createElement("li", null, "Place dominoes on a plain, high-contrast surface."),
            React.createElement("li", null, "Keep camera parallel to the table and reduce glare."),
            React.createElement("li", null, "Tap Scan Dominoes and verify the overlay boxes + counts.")
          )
        )
      ),
      React.createElement("canvas", { ref: captureRef, hidden: true })
    );
  }

  ReactDOM.createRoot(document.getElementById("root")).render(React.createElement(App));
})();
