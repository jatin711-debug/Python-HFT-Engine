/**
 * WatchlistWidget - Mini price cards (Compact)
 */

import { useSelector, useDispatch } from 'react-redux';
import { TrendingUp, TrendingDown } from 'lucide-react';
import { selectCoins, selectActiveSymbol, setActiveSymbol, selectMarketType } from '../../store/tradingSlice';

export const WatchlistWidget = ({ sendMessage }) => {
    const dispatch = useDispatch();
    const coins = useSelector(selectCoins);
    const activeSymbol = useSelector(selectActiveSymbol);
    const marketType = useSelector(selectMarketType);

    const symbols = Object.keys(coins).slice(0, 5); // Limit to 5

    const handleSelect = (symbol) => {
        dispatch(setActiveSymbol(symbol));
        if (sendMessage) {
            sendMessage({ action: 'set_active', symbol });
        }
    };

    return (
        <div className="glass-card">
            <div className="card-header py-2 px-3">
                <div className="flex items-center justify-between w-full">
                    <span className="text-xs font-semibold">Watchlist</span>
                    <span className="badge badge-primary text-[10px]">{symbols.length}</span>
                </div>
            </div>

            <div className="px-2 pb-2 max-h-[140px] overflow-y-auto">
                {symbols.map(symbol => {
                    const data = coins[symbol];
                    const price = data?.price || 0;
                    const change = data?.price_change_pct || 0;
                    const isPositive = change >= 0;
                    const displaySymbol = marketType === 'stocks' ? symbol : symbol.replace('USDT', '');

                    return (
                        <button
                            key={symbol}
                            onClick={() => handleSelect(symbol)}
                            className={`w-full flex items-center justify-between p-2 rounded text-xs transition-all ${symbol === activeSymbol ? 'bg-blue-500/20' : 'hover:bg-white/5'
                                }`}
                        >
                            <span className="font-medium">{displaySymbol}</span>
                            <div className="flex items-center gap-2">
                                <span className="font-mono">${price.toFixed(2)}</span>
                                <span className={`flex items-center ${isPositive ? 'text-green-500' : 'text-red-500'}`}>
                                    {isPositive ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                                    {Math.abs(change).toFixed(1)}%
                                </span>
                            </div>
                        </button>
                    );
                })}
            </div>
        </div>
    );
};
