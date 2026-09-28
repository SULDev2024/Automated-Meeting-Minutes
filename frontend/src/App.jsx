import { useEffect, useState } from "react";
import "./App.css";
import demoMeeting from "./data/demoMeeting.json";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [backendOnline, setBackendOnline] = useState(null);

  useEffect(() => {
    const checkBackend = async () => {
      try {
        const response = await fetch("/api/health");
        if (!response.ok) throw new Error();

        const data = await response.json();
        setBackendOnline(data.status === "ok");
      } catch {
        setBackendOnline(false);
      }
    };

    checkBackend();
  }, []);

  const selectFile = (selectedFile) => {
    if (!selectedFile) return;

    setFile(selectedFile);
    setResult(null);
    setError("");
  };

  const handleDrop = (event) => {
    event.preventDefault();
    selectFile(event.dataTransfer.files?.[0]);
  };

  const loadDemo = () => {
    setFile(null);
    setError("");
    setResult({
      ...demoMeeting,
      isDemo: true,
      // Bundled copies of this saved run's exports (frontend/public/demo),
      // so demo downloads work on any laptop without the backend's files.
      exports: {
        pdf: "/demo/meeting-demo-minutes.pdf",
        docx: "/demo/meeting-demo-minutes.docx",
      },
    });
  };

  const downloadUrl = (path) => (result?.isDemo ? path : `/api${path}`);

  const analyzeMeeting = async () => {
    if (!file || loading) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("/api/meetings/analyze", {
        method: "POST",
        body: formData,
      });

      let data;

      try {
        data = await response.json();
      } catch {
        throw new Error(
          response.ok
            ? "The server returned an invalid response."
            : "Could not connect to the meeting analysis service."
        );
      }

      if (!response.ok) {
        throw new Error(
          data?.detail ||
          data?.message ||
          "Meeting analysis failed."
        );
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "Could not connect to the backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="navbar">
        <div className="brand">
          <div className="brandIcon">M</div>
          <div>
            <h2>MeetingAI</h2>
            <span>Automated Meeting Intelligence</span>
          </div>
        </div>

        <div className={`status ${backendOnline === false ? "offline" : ""}`}>
          <span className="statusDot"></span>
          {backendOnline === null
            ? "Checking System..."
            : backendOnline
              ? "System Online"
              : "System Offline"}
        </div>
      </header>

      <main className="container">
        <section className="hero">
          <div className="badge">AI-POWERED MEETING ANALYSIS</div>

          <h1>
            Turn meetings into
            <span> clear action.</span>
          </h1>

          <p>
            Upload your meeting recording and automatically generate a
            structured summary, action items, deadlines, speaker transcript,
            and professional meeting minutes.
          </p>
        </section>

        <section className="uploadCard">
          <div className="uploadHeader">
            <div>
              <h2>Analyze a Meeting</h2>
              <p>Upload a recording to generate intelligent meeting minutes.</p>
            </div>

            <span className="supported">MP3 · MP4 · WAV · M4A · WEBM</span>
          </div>

          <label
            className="dropZone"
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
          >
            <div className="uploadIcon">↑</div>

            {file ? (
              <>
                <h3>{file.name}</h3>
                <p>Ready for analysis</p>
              </>
            ) : (
              <>
                <h3>Drop your meeting recording here</h3>
                <p>or click to browse your files</p>
              </>
            )}

            <input
              type="file"
              accept=".mp3,.mp4,.wav,.m4a,.webm"
              onChange={(e) => selectFile(e.target.files[0])}
              hidden
            />
          </label>

          <div className="uploadFooter">
            <p>
              Russian · Kazakh · Mixed-language meetings supported
            </p>

            <div className="footerActions">
              <button
                type="button"
                className="demoButton"
                onClick={loadDemo}
                disabled={loading}
              >
                Load Demo
              </button>

              <button
                disabled={!file || loading}
                onClick={analyzeMeeting}
              >
                {loading ? "Analyzing..." : "Analyze Meeting"}
                {!loading && <span>→</span>}
              </button>
            </div>
          </div>
        </section>

        {error && (
          <div className="apiMessage errorMessage">
            {error}
          </div>
        )}

        {result && (
          <section className="results">
            <div className="resultsHeader">
              <div>
                <span className={`resultBadge ${result.isDemo ? "demoBadge" : ""}`}>
                  {result.isDemo ? "DEMO — SAVED RESULT" : "ANALYSIS COMPLETE"}
                </span>
                <h2>Meeting Intelligence</h2>
                <p>{result.action_items?.length || 0} action items identified</p>
                {result.isDemo && (
                  <p className="demoNote">
                    Saved output from a verified live run of{" "}
                    {result.filename || "a sample meeting"}. No AI call was made.
                  </p>
                )}
              </div>

              <div className="downloadButtons">
                {result.exports?.pdf && (
                  <a
                    href={downloadUrl(result.exports.pdf)}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Download PDF
                  </a>
                )}

                {result.exports?.docx && (
                  <a href={downloadUrl(result.exports.docx)} download>
                    Download DOCX
                  </a>
                )}
              </div>
            </div>

            <div className="summaryCard">
              <span className="sectionLabel">MEETING SUMMARY</span>
              <p>
                {typeof result.summary === "string"
                  ? result.summary
                  : result.summary?.summary ||
                    result.summary?.text ||
                    "Summary generated successfully."}
              </p>
            </div>

            <div className="actionsSection">
              <div className="sectionTitle">
                <h2>Action Items</h2>
                <span>{result.action_items?.length || 0}</span>
              </div>

              <div className="actionList">
                {result.action_items?.map((action, index) => (
                  <div className="actionCard" key={index}>
                    <div className="actionNumber">
                      {String(index + 1).padStart(2, "0")}
                    </div>

                    <div className="actionContent">
                      <h3>{action.task || "Action item"}</h3>

                      <div className="actionMeta">
                        <span>
                          <strong>Owner</strong>
                          {action.owner || "Not identified"}
                        </span>

                        <span>
                          <strong>Deadline</strong>
                          {action.deadline || "Not specified"}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <details className="transcriptCard">
              <summary>View Full Transcript</summary>

              <div className="transcriptContent">
                {result.diarization?.segments?.map((segment, index) => (
                  <div className="transcriptLine" key={index}>
                    <strong>{segment.speaker || `Speaker ${index + 1}`}</strong>
                    <p>{segment.text}</p>
                  </div>
                ))}
              </div>
            </details>
          </section>
        )}

        <section className="features">
          <div className="feature">
            <div className="featureIcon">01</div>
            <div>
              <h3>Speaker Recognition</h3>
              <p>Separate speakers and understand who said what.</p>
            </div>
          </div>

          <div className="feature">
            <div className="featureIcon">02</div>
            <div>
              <h3>Action Tracking</h3>
              <p>Automatically identify tasks, owners and deadlines.</p>
            </div>
          </div>

          <div className="feature">
            <div className="featureIcon">03</div>
            <div>
              <h3>Instant Minutes</h3>
              <p>Export professional meeting minutes as PDF or DOCX.</p>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
