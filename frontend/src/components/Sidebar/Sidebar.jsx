import "./Sidebar.css";

function Sidebar() {
  return (
    <aside className="sidebar">

      <h2 className="sidebar-title">
        EEG Status
      </h2>

      <div className="status-card">
        <span>Connection</span>
        <strong>Connected</strong>
      </div>

      <div className="status-card">
        <span>Model</span>
        <strong>CSP + LDA</strong>
      </div>

      <div className="status-card">
        <span>Sampling Rate</span>
        <strong>160 Hz</strong>
      </div>

      <div className="status-card">
        <span>Channels</span>
        <strong>64</strong>
      </div>

      <div className="status-card">
        <span>Latency</span>
        <strong>18 ms</strong>
      </div>

      <div className="status-card">
        <span>Subject</span>
        <strong>S001</strong>
      </div>

    </aside>
  );
}

export default Sidebar;