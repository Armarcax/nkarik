import { useRef, useState, useEffect, useCallback } from "react";

const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

const STYLES = [
  { id: "cartoon ghibli anime style",   label: "Ghibli",     bg: "#e8f5e9", accent: "#43a047", icon: "🌿" },
  { id: "pixar 3d cartoon style",       label: "Pixar",      bg: "#e3f2fd", accent: "#1e88e5", icon: "🎬" },
  { id: "cute watercolor illustration", label: "Watercolor", bg: "#fce4ec", accent: "#e91e63", icon: "🎨" },
  { id: "comic book pop art style",     label: "Comic",      bg: "#fff9c4", accent: "#f9a825", icon: "💥" },
  { id: "cute kawaii sticker art",      label: "Kawaii",     bg: "#f3e5f5", accent: "#8e24aa", icon: "🌸" },
];

const COLORS = [
  "#212121", "#f44336", "#ff9800", "#ffeb3b",
  "#4caf50", "#2196f3", "#9c27b0", "#ffffff",
];

const BRUSH_SIZES = [
  { size: 3,  label: "Fine" },
  { size: 8,  label: "Medium" },
  { size: 16, label: "Bold" },
  { size: 28, label: "Thick" },
];

const MAGIC_MESSAGES = [
  "Mixing magic colors… 🎨",
  "Adding sparkles… ✨",
  "Talking to the art fairies… 🧚",
  "Making your drawing fly… 🦋",
  "Pouring in imagination… 🌈",
  "Almost ready… hold on! 🌟",
];

const COMPLIMENTS = [
  "Wow, you're an amazing artist! 🌟",
  "That looks incredible! 🎉",
  "You made something magical! ✨",
  "This is your masterpiece! 🏆",
  "You have a superpower! 🦸",
];

const MAX_HISTORY = 20;

function getPos(e, canvas) {
  const rect = canvas.getBoundingClientRect();
  const sx = canvas.width / rect.width;
  const sy = canvas.height / rect.height;
  const src = e.touches ? e.touches[0] : e;
  return {
    x: (src.clientX - rect.left) * sx,
    y: (src.clientY - rect.top) * sy,
  };
}

function randomFrom(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

function Particles() {
  const items = ["⭐", "✨", "🌟", "💫", "🎨", "🌈"];
  return (
    <div className="particles" aria-hidden="true">
      {[...Array(12)].map((_, i) => (
        <span key={i} className="particle"
          style={{ "--x": `${Math.random() * 100}%`, "--delay": `${Math.random() * 4}s`, "--dur": `${3 + Math.random() * 4}s` }}>
          {items[i % items.length]}
        </span>
      ))}
    </div>
  );
}

function MagicLoader() {
  const [msgIdx, setMsgIdx] = useState(0);
  const [dots, setDots] = useState(0);
  useEffect(() => {
    const t1 = setInterval(() => setMsgIdx((i) => (i + 1) % MAGIC_MESSAGES.length), 1800);
    const t2 = setInterval(() => setDots((d) => (d + 1) % 4), 400);
    return () => { clearInterval(t1); clearInterval(t2); };
  }, []);
  return (
    <div className="magic-loader">
      <div className="wand-wrap">
        <div className="wand">
          <div className="wand-tip" />
          {[...Array(6)].map((_, i) => (
            <div key={i} className="wand-spark" style={{ "--i": i }} />
          ))}
        </div>
      </div>
      <p className="magic-msg">{MAGIC_MESSAGES[msgIdx]}{".".repeat(dots)}</p>
      <div className="progress-track"><div className="progress-bar" /></div>
    </div>
  );
}

export default function App() {
  const canvasRef  = useRef(null);
  const drawing    = useRef(false);
  const lastPos    = useRef(null);
  const historyRef = useRef([]);
  const historyIdx = useRef(-1);

  const [brushSize,     setBrushSize]     = useState(8);
  const [brushColor,    setBrushColor]    = useState("#212121");
  const [tool,          setTool]          = useState("pen");
  const [selectedStyle, setSelectedStyle] = useState(0);
  const [strength,      setStrength]      = useState(0.65);
  const [resultUrl,     setResultUrl]     = useState(null);
  const [loading,       setLoading]       = useState(false);
  const [error,         setError]         = useState(null);
  const [screen,        setScreen]        = useState("draw");
  const [compliment,    setCompliment]    = useState("");
  const [showHint,      setShowHint]      = useState(true);
  const [canUndo,       setCanUndo]       = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    saveHistory();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function saveHistory() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const data = canvas.toDataURL();
    historyRef.current = historyRef.current.slice(0, historyIdx.current + 1);
    historyRef.current.push(data);
    if (historyRef.current.length > MAX_HISTORY) historyRef.current.shift();
    historyIdx.current = historyRef.current.length - 1;
    setCanUndo(historyIdx.current > 0);
  }

  function undo() {
    if (historyIdx.current <= 0) return;
    historyIdx.current--;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const img = new Image();
    img.src = historyRef.current[historyIdx.current];
    img.onload = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);
      setCanUndo(historyIdx.current > 0);
    };
  }

  const startDraw = useCallback((e) => {
    e.preventDefault();
    drawing.current = true;
    setShowHint(false);
    lastPos.current = getPos(e, canvasRef.current);
  }, []);

  const endDraw = useCallback(() => {
    if (!drawing.current) return;
    drawing.current = false;
    lastPos.current = null;
    saveHistory();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const drawMove = useCallback((e) => {
    if (!drawing.current) return;
    e.preventDefault();
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const pos = getPos(e, canvas);
    ctx.globalCompositeOperation = tool === "eraser" ? "destination-out" : "source-over";
    ctx.strokeStyle = tool === "eraser" ? "rgba(0,0,0,1)" : brushColor;
    ctx.lineWidth   = tool === "eraser" ? brushSize * 2.5 : brushSize;
    ctx.lineCap  = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(lastPos.current.x, lastPos.current.y);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    lastPos.current = pos;
  }, [brushColor, brushSize, tool]);

  const clearCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.globalCompositeOperation = "source-over";
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setShowHint(true);
    setError(null);
    saveHistory();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleGenerate = async () => {
    setError(null);
    setLoading(true);
    setCompliment(randomFrom(COMPLIMENTS));
    try {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");
      ctx.globalCompositeOperation = "source-over";
      const imageData = canvas.toDataURL("image/png");
      const style = STYLES[selectedStyle];
      const res = await fetch(`${API_URL}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageData, style: style.id, strength }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail ?? `Server error ${res.status}`);
      }
      const json = await res.json();
      setResultUrl(json.result);
      setScreen("result");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSave = () => {
    if (!resultUrl) return;
    const a = document.createElement("a");
    a.href = resultUrl;
    a.download = `nkarik-art-${Date.now()}.png`;
    a.click();
  };

  const handleDrawAgain = () => {
    setScreen("draw");
    setResultUrl(null);
    setError(null);
    clearCanvas();
  };

  // Result screen
  if (screen === "result") {
    return (
      <div className="app result-screen">
        <Particles />
        <div className="result-header">
          <h1 className="logo">Nkarik</h1>
          <p className="compliment">{compliment}</p>
        </div>
        <div className="result-card">
          {resultUrl && <img src={resultUrl} alt="Your AI artwork" className="result-img reveal-img" />}
          <div className="result-badge">{STYLES[selectedStyle].icon} {STYLES[selectedStyle].label} style</div>
        </div>
        <div className="result-actions">
          <button className="btn-secondary" onClick={handleSave}><span>💾</span> Save</button>
          <button className="btn-secondary" onClick={() => navigator.share?.({ title: "My Nkarik art!", url: resultUrl })}><span>📤</span> Share</button>
        </div>
        <button className="btn-magic" onClick={handleDrawAgain}>✏️ Draw something new!</button>
      </div>
    );
  }

  // Loading screen
  if (loading) {
    return (
      <div className="app loading-screen">
        <h1 className="logo">Nkarik</h1>
        <MagicLoader />
      </div>
    );
  }

  // Draw screen
  return (
    <div className="app draw-screen">
      <header className="app-header">
        <h1 className="logo">Nkarik 🎨</h1>
        <p className="tagline">Draw anything — AI makes it magical</p>
      </header>

      <div className="canvas-wrap">
        {showHint && <div className="canvas-hint" aria-hidden="true"><span>✏️ Start drawing here!</span></div>}
        <canvas ref={canvasRef} width={512} height={512} className={`draw-canvas tool-${tool}`}
          onMouseDown={startDraw} onMouseMove={drawMove} onMouseUp={endDraw} onMouseLeave={endDraw}
          onTouchStart={startDraw} onTouchMove={drawMove} onTouchEnd={endDraw} />
      </div>

      <div className="toolbar card">
        <div className="toolbar-section">
          <span className="section-label">Colors</span>
          <div className="color-row">
            {COLORS.map((c) => (
              <button key={c} className={`color-swatch ${brushColor === c && tool === "pen" ? "active" : ""}`}
                style={{ "--clr": c }} onClick={() => { setBrushColor(c); setTool("pen"); }} aria-label={c} />
            ))}
          </div>
        </div>

        <div className="toolbar-section">
          <span className="section-label">Brush size</span>
          <div className="brush-row">
            {BRUSH_SIZES.map(({ size, label }) => (
              <button key={size}
                className={`brush-btn ${brushSize === size && tool === "pen" ? "active" : ""}`}
                onClick={() => { setBrushSize(size); setTool("pen"); }} title={label}>
                <span className="brush-dot" style={{ width: Math.min(size, 24), height: Math.min(size, 24) }} />
              </button>
            ))}
          </div>
        </div>

        <div className="toolbar-section tool-actions">
          <button className={`action-btn ${tool === "eraser" ? "active" : ""}`}
            onClick={() => setTool(tool === "eraser" ? "pen" : "eraser")} title="Eraser">
            🧹 Eraser
          </button>
          <button className={`action-btn ${!canUndo ? "disabled" : ""}`}
            onClick={undo} title="Undo" disabled={!canUndo}>
            ↩️ Undo
          </button>
          <button className="action-btn danger" onClick={clearCanvas} title="Clear all">
            🗑️ Clear
          </button>
        </div>
      </div>

      <div className="style-picker card">
        <span className="section-label">Art style</span>
        <div className="style-row">
          {STYLES.map((st, i) => (
            <button key={st.id} className={`style-btn ${selectedStyle === i ? "active" : ""}`}
              style={{ "--bg": st.bg, "--ac": st.accent }} onClick={() => setSelectedStyle(i)}>
              <span className="style-icon">{st.icon}</span>
              <span className="style-label">{st.label}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="creativity-card card">
        <div className="creativity-header">
          <span className="section-label">Imagination level</span>
          <span className="creativity-val">{Math.round(strength * 100)}%</span>
        </div>
        <input type="range" className="creativity-slider"
          min={0.3} max={0.9} step={0.05} value={strength}
          onChange={(e) => setStrength(parseFloat(e.target.value))} />
        <div className="creativity-labels">
          <span>📐 Stay close</span>
          <span>🚀 Go wild!</span>
        </div>
      </div>

      {error && <div className="error-box">⚠️ {error}</div>}

      <button className="btn-magic" onClick={handleGenerate}>
        ✨ Կենդանացնել — Make it magical!
      </button>
    </div>
  );
}
