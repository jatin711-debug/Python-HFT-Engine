export const Card = ({ children, className = '' }) => (
    <div className={`glass-card rounded-xl p-5 ${className}`}>
        {children}
    </div>
);

export const Badge = ({ children, type = 'neutral' }) => {
    const styles = {
        neutral: 'bg-gray-800 text-gray-300 border-gray-700',
        success: 'bg-emerald-900/30 text-emerald-400 border-emerald-800',
        danger: 'bg-red-900/30 text-red-400 border-red-800',
        warning: 'bg-amber-900/30 text-amber-400 border-amber-800',
        active: 'bg-blue-900/30 text-blue-400 border-blue-800',
    };
    return (
        <span className={`px-2.5 py-0.5 rounded-full text-xs font-medium border ${styles[type]}`}>
            {children}
        </span>
    );
};
