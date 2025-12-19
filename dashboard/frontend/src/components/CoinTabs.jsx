import { useDispatch } from 'react-redux';
import { Power } from 'lucide-react';
import { setActiveSymbol } from '../store/tradingSlice';

export const CoinTabs = ({ coins, activeSymbol, sendMessage, data }) => {
    const dispatch = useDispatch();

    const handleSelect = (symbol) => {
        dispatch(setActiveSymbol(symbol));
        sendMessage({ action: 'set_active', symbol });
    };

    return (
        <div className="flex items-center gap-2 bg-bg-secondary p-1 rounded-lg border border-border">
            {coins.map((symbol) => {
                const coinData = data?.coins?.[symbol] || {};
                const autoTradeEnabled = coinData.auto_trade_enabled !== false;

                return (
                    <div key={symbol} className="flex items-center gap-1">
                        <button
                            onClick={() => handleSelect(symbol)}
                            className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${activeSymbol === symbol
                                    ? 'bg-bg-card text-white shadow-lg border border-gray-600'
                                    : 'text-gray-400 hover:text-white hover:bg-white/5'
                                }`}
                        >
                            {symbol.replace('USDT', '')}
                        </button>

                        {/* Auto-trade toggle */}
                        <button
                            onClick={() => sendMessage({ action: 'toggle_auto_trade', symbol })}
                            className={`p-1.5 rounded-md transition-all ${autoTradeEnabled
                                    ? 'bg-emerald-600/20 text-emerald-400 hover:bg-emerald-600/30'
                                    : 'bg-gray-700 text-gray-500 hover:bg-gray-600'
                                }`}
                            title={`Auto-trade ${autoTradeEnabled ? 'ON' : 'OFF'}`}
                        >
                            <Power className="w-3 h-3" />
                        </button>
                    </div>
                );
            })}
            <button className="px-3 py-1.5 text-gray-500 hover:text-gray-300 text-xs">+ ADD</button>
        </div>
    );
};
