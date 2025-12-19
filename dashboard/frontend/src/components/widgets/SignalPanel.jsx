/**
 * SignalPanel Widget - Live trading signals (Compact)
 */

import { useSelector } from 'react-redux';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { selectCoins } from '../../store/tradingSlice';

const SignalRow = ({ symbol, signal }) => {
    const direction = signal?.signal_direction || 'HOLD';
    const confidence = (signal?.signal_confidence || 0) * 100;
    const price = signal?.price || 0;

    const getBadgeClass = () => {
        if (direction === 'BUY') return 'badge-success';
        if (direction === 'SELL') return 'badge-danger';
        return 'badge-warning';
    };

    return (
        <div className="flex items-center justify-between py-2 border-b last:border-0"
            style={{ borderColor: 'var(--border-subtle)' }}>
            <div className="flex items-center gap-2">
                <span className="font-medium text-xs">{symbol}</span>
                <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
                    ${price.toFixed(2)}
                </span>
            </div>
            <div className="flex items-center gap-2">
                <span className={`badge text-[10px] ${getBadgeClass()}`}>{direction}</span>
                <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
                    {confidence.toFixed(0)}%
                </span>
            </div>
        </div>
    );
};

export const SignalPanel = () => {
    const coins = useSelector(selectCoins);
    const symbols = Object.keys(coins).slice(0, 4); // Show top 4 to save space

    return (
        <div className="glass-card">
            <div className="card-header py-2 px-3">
                <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold">Live Signals</span>
                    <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                </div>
            </div>

            <div className="px-3 pb-2">
                {symbols.length > 0 ? (
                    symbols.map(symbol => (
                        <SignalRow
                            key={symbol}
                            symbol={symbol.replace('USDT', '')}
                            signal={coins[symbol]}
                        />
                    ))
                ) : (
                    <div className="py-2 text-center text-xs" style={{ color: 'var(--text-muted)' }}>
                        Waiting for signals...
                    </div>
                )}
            </div>
        </div>
    );
};
