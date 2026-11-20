
import { useEffect, useState } from "react";
import axios from "axios";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from "recharts";
import {
  Activity,
  ShieldAlert,
  ShieldCheck,
  AlertTriangle,
  RefreshCw,
  Send,
  Trash2,
  Cpu,
} from "lucide-react";
import "./App.css";

const API = "http://127.0.0.1:8000";
const SESSION_ID = "dashboard-demo";

function App() {
  const [goal, setGoal] = useState(
    "Find the cheapest laptop under 60000 rupees with 16GB RAM"
  );
  const [action, setAction] = useState("");
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState({
    total_actions: 0,
    status_counts: {},
    average_drift_score: 0,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [latest, setLatest] = useState(null);

  async function refreshDashboard() {
    try {
      const [historyResponse, statsResponse] = await Promise.all([
        axios.get(`${API}/history`, {
          params: { session_id: SESSION_ID },
        }),
        axios.get(`${API}/stats`, {
          params: { session_id: SESSION_ID },
        }),
      ]);

      const records = historyResponse.data.history || [];
      setHistory(records);
      setStats(statsResponse.data);
      setLatest(records.length ? records[records.length - 1] : null);
      setError("");
    } catch (err) {
      setError(
        "Cannot connect to the backend. Make sure FastAPI is running on port 8000."
      );
    }
  }

  useEffect(() => {
    refreshDashboard();
  }, []);

  async function analyzeAction(event) {
    event.preventDefault();

    if (!action.trim() || !goal.trim()) return;

    setLoading(true);
    setError("");

    try {
      await axios.post(`${API}/analyze`, {
        session_id: SESSION_ID,
        goal,
        action,
        action_type: "general",
      });

      setAction("");
      await refreshDashboard();
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        "Action analysis failed. Check the backend terminal."
      );
    } finally {
      setLoading(false);
    }
  }

  async function resetSession() {
    if (!window.confirm("Reset this monitoring session and clear its history?")) {
      return;
    }

    try {
      await axios.post(`${API}/reset`, {
        session_id: SESSION_ID,
      });

      setHistory([]);
      setLatest(null);
      setStats({
        total_actions: 0,
        status_counts: {},
        average_drift_score: 0,
      });
      setError("");
    } catch {
      setError("Could not reset the monitoring session.");
    }
  }

  const chartData = history.map((record, index) => ({
    action: `Action ${index + 1}`,
    drift: Number(record.drift_score.toFixed(3)),
    similarity: Number(record.similarity.toFixed(3)),
  }));

  const status = latest?.status || "NO DATA";

  function statusClass(value) {
    return value.toLowerCase().replaceAll(" ", "-");
  }

  return (
    <div className="dashboard">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon"><Cpu size={25} /></div>
          <div>
            <h2>IntentDrift</h2>
            <span>AGENT MONITORING</span>
          </div>
        </div>

        <nav>
          <a className="nav-active"><Activity size={18} /> Dashboard</a>
          <a href="#analysis"><Send size={18} /> Analyze Action</a>
          <a href="#history"><RefreshCw size={18} /> Action History</a>
        </nav>

        <div className="sidebar-bottom">
          <div className="connection">
            <span className="online-dot"></span>
            API CONNECTION
            <strong>LOCAL SERVER</strong>
          </div>
          <p>IntentDrift v1.0</p>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <p className="eyebrow">AUTONOMOUS AI SAFETY SYSTEM</p>
            <h1>Monitoring Dashboard</h1>
            <p className="subtitle">
              Track agent behaviour and detect deviation from original goals.
            </p>
          </div>
          <button className="refresh-button" onClick={refreshDashboard}>
            <RefreshCw size={16} /> Refresh
          </button>
        </header>

        {error && <div className="error-banner">{error}</div>}

        <section className="goal-panel">
          <div>
            <span className="section-label">ORIGINAL USER GOAL</span>
            <p>{goal}</p>
          </div>
          <span className="session-tag">SESSION: {SESSION_ID}</span>
        </section>

        <section className="metric-grid">
          <MetricCard
            icon={<Activity />}
            label="Total Actions"
            value={stats.total_actions}
            note="Actions analyzed"
          />
          <MetricCard
            icon={<ShieldCheck />}
            label="Aligned Actions"
            value={stats.status_counts?.ALIGNED || 0}
            note="Within goal boundaries"
            color="green"
          />
          <MetricCard
            icon={<AlertTriangle />}
            label="Warnings"
            value={stats.status_counts?.WARNING || 0}
            note="Potential goal deviation"
            color="yellow"
          />
          <MetricCard
            icon={<ShieldAlert />}
            label="Average Drift Score"
            value={Number(stats.average_drift_score || 0).toFixed(3)}
            note="Across analyzed actions"
            color="purple"
          />
        </section>

        <section className="content-grid">
          <div className="panel chart-panel">
            <div className="panel-heading">
              <div>
                <span className="section-label">TEMPORAL ANALYSIS</span>
                <h2>Drift Score Trend</h2>
              </div>
              <span className="live-tag">LIVE DATA</span>
            </div>

            {chartData.length ? (
              <ResponsiveContainer width="100%" height={280}>
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="driftFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#a78bfa" stopOpacity={0.4} />
                      <stop offset="100%" stopColor="#a78bfa" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#293044" />
                  <XAxis dataKey="action" stroke="#8992a8" />
                  <YAxis domain={[0, 1]} stroke="#8992a8" />
                  <Tooltip contentStyle={{ background: "#171b28", border: "1px solid #343b50" }} />
                  <Area
                    type="monotone"
                    dataKey="drift"
                    stroke="#a78bfa"
                    fill="url(#driftFill)"
                    strokeWidth={3}
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="empty-chart">Analyze an action to generate your first chart.</div>
            )}
          </div>

          <div className="panel status-panel">
            <span className="section-label">CURRENT AGENT STATUS</span>
            <div className={`status-symbol ${statusClass(status)}`}>
              {status === "ALIGNED" ? <ShieldCheck size={38} /> :
                status === "NO DATA" ? <Activity size={38} /> :
                <AlertTriangle size={38} />}
            </div>
            <h2 className={`status-text ${statusClass(status)}`}>{status}</h2>
            <p>Latest classification</p>

            <div className="score-box">
              <span>Latest Drift Score</span>
              <strong>{latest ? latest.drift_score.toFixed(4) : "—"}</strong>
            </div>

            <div className="intervention">
              <span>RECOMMENDED INTERVENTION</span>
              <strong>{latest?.intervention || "Awaiting agent actions"}</strong>
            </div>
          </div>
        </section>

        <section className="content-grid lower-grid">
          <div className="panel" id="analysis">
            <div className="panel-heading">
              <div>
                <span className="section-label">TEST YOUR AGENT</span>
                <h2>Analyze New Action</h2>
              </div>
              <Send size={20} />
            </div>

            <form onSubmit={analyzeAction}>
              <label>Original Goal</label>
              <textarea
                value={goal}
                onChange={(e) => setGoal(e.target.value)}
                rows={3}
                required
              />

              <label>Agent Action</label>
              <textarea
                value={action}
                onChange={(e) => setAction(e.target.value)}
                placeholder="Enter what the AI agent is currently doing..."
                rows={3}
                required
              />

              <button className="primary-button" type="submit" disabled={loading}>
                <Send size={17} />
                {loading ? "Analyzing..." : "Analyze Agent Action"}
              </button>
            </form>
          </div>

          <div className="panel">
            <div className="panel-heading">
              <div>
                <span className="section-label">SEMANTIC COMPARISON</span>
                <h2>Latest Similarity</h2>
              </div>
              <Activity size={20} />
            </div>

            <div className="similarity-value">
              {latest ? `${(latest.similarity * 100).toFixed(1)}%` : "—"}
            </div>
            <p className="subtitle">Similarity between original goal and latest action</p>

            <div className="progress-track">
              <div
                className="progress-fill"
                style={{ width: `${(latest?.similarity || 0) * 100}%` }}
              />
            </div>

            <div className="comparison-details">
              <span>Instantaneous Drift</span>
              <strong>
                {latest ? latest.instantaneous_drift.toFixed(4) : "—"}
              </strong>
            </div>
            <div className="comparison-details">
              <span>Persistence</span>
              <strong>{latest ? latest.persistence.toFixed(2) : "—"}</strong>
            </div>
          </div>
        </section>

        <section className="panel history-panel" id="history">
          <div className="panel-heading">
            <div>
              <span className="section-label">MONITORING RECORDS</span>
              <h2>Agent Action History</h2>
            </div>
            <button className="reset-button" onClick={resetSession}>
              <Trash2 size={15} /> Reset Session
            </button>
          </div>

          {history.length ? (
            <div className="table-wrapper">
              <table>
                <thead>
                  <tr>
                    <th>Action</th>
                    <th>Similarity</th>
                    <th>Drift Score</th>
                    <th>Status</th>
                    <th>Intervention</th>
                  </tr>
                </thead>
                <tbody>
                  {history.slice().reverse().map((record, index) => (
                    <tr key={`${record.timestamp}-${index}`}>
                      <td className="action-cell">{record.action}</td>
                      <td>{record.similarity.toFixed(3)}</td>
                      <td>{record.drift_score.toFixed(3)}</td>
                      <td>
                        <span className={`status-pill ${statusClass(record.status)}`}>
                          {record.status}
                        </span>
                      </td>
                      <td>{record.intervention}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="empty-chart">No actions recorded yet.</div>
          )}
        </section>

        <footer>
          INTENTDRIFT · GOAL DRIFT DETECTION IN AUTONOMOUS AI AGENTS
        </footer>
      </main>
    </div>
  );
}

function MetricCard({ icon, label, value, note, color = "blue" }) {
  return (
    <div className="metric-card">
      <div className={`metric-icon ${color}`}>{icon}</div>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{note}</small>
    </div>
  );
}

export default App;
