export const CoinTabs = ({ coins, activeSymbol, onSelect }) => (
    <div className="flex items-center gap-2 bg-bg-secondary p-1 rounded-lg border border-border">
        {coins.map(symbol => (
            <button
                key={symbol}
                onClick={() => onSelect(symbol)}
                className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${activeSymbol === symbol
                        ? 'bg-bg-card text-white shadow-lg border border-gray-600'
                        : 'text-gray-400 hover:text-white hover:bg-white/5'
                    }`}
            >
                {symbol.replace('USDT', '')}
            </button>
        ))}
        <button className="px-3 py-1.5 text-gray-500 hover:text-gray-300 text-xs">+ ADD</button>
    </div>
);
