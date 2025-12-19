/**
 * QuickTrade Widget - One-click trading panel (Compact)
 */

import { useState } from 'react';
import { useSelector } from 'react-redux';
import { TrendingUp, TrendingDown } from 'lucide-react';
import { selectActiveCoinData, selectActiveSymbol, selectMarketType } from '../../store/tradingSlice';

export const QuickTrade = ({ sendMessage }) => {
    const activeSymbol = useSelector(selectActiveSymbol);
    const activeCoin = useSelector(selectActiveCoinData);
    const marketType = useSelector(selectMarketType);

    const [amount, setAmount] = useState('100');

    const price = activeCoin?.price || 0;
    const displaySymbol = marketType === 'stocks'
        ? activeSymbol
        : activeSymbol?.replace('USDT', '');

    const handleTrade = (side) => {
        if (sendMessage) {
            sendMessage({ action: 'toggle_auto_trade', symbol: activeSymbol });
        }
    };

    return (
        <div className="glass-card">
            <div className="card-header py-2 px-3">
                <div className="flex items-center justify-between w-full">
                    <span className="text-xs font-semibold">Quick Trade</span>
                    <span className="badge badge-primary text-[10px]">{displaySymbol}</span>
                </div>
            </div>

            <div className="p-3 space-y-3">
                {/* Current Price */}
                <div className="text-center">
                    <span className="text-lg font-bold font-mono">${price.toFixed(2)}</span>
                </div>

                {/* Amount Input */}
                <div>
                    <input
                        type="number"
                        value={amount}
                        onChange={(e) => setAmount(e.target.value)}
                        className="input text-sm py-2"
                        placeholder="Amount (USD)"
                    />
                </div>

                {/* Buy/Sell Buttons */}
                <div className="grid grid-cols-2 gap-2">
                    <button
                        onClick={() => handleTrade('BUY')}
                        className="btn btn-success py-2 text-sm"
                    >
                        <TrendingUp className="w-3 h-3" />
                        BUY
                    </button>
                    <button
                        onClick={() => handleTrade('SELL')}
                        className="btn btn-danger py-2 text-sm"
                    >
                        <TrendingDown className="w-3 h-3" />
                        SELL
                    </button>
                </div>
            </div>
        </div>
    );
};
