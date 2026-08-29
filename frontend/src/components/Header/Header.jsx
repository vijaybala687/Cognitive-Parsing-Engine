import "./Header.css";

function Header() {
  return (
    <header className="header">

      <div className="header-left">

        <div className="logo-circle">
          🧠
        </div>

        <div>
          <h1>Cognitive Parsing Engine</h1>
          <p>Brain Computer Interface</p>
        </div>

      </div>


      <div className="header-right">

        <div className="status">
          <span className="status-dot"></span>
          EEG Connected
        </div>

      </div>

    </header>
  );
}

export default Header;