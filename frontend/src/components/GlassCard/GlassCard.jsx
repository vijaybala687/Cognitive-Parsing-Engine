import "./GlassCard.css";

function GlassCard({ children }) {
    return (
        <div className="glassCard">
            {children}
        </div>
    );
}

export default GlassCard;