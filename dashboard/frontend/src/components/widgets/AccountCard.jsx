/**
 * AccountCard Widget - Balance and P&L summary (Compact)
 */

import { useSelector } from 'react-redux';
import { TrendingUp, TrendingDown, Trophy, Target } from 'lucide-react';
import { selectGlobalStats } from '../../store/tradingSlice';

export const AccountCard = () => {
    const stats = useSelector(selectGlobalStats);

    const netPnl = stats?.netPnl || 0;
    const winRate = stats?.winRate || 0;
    const totalTrades = stats?.totalTrades || 0;
    const openPositions = stats?.openPositions || 0;
    const isProfit = netPnl >= 0;

    return (
        <div className="glass-card">
            <div className="card-header py-2 px-3">
                <span className="text-xs font-semibold">Account Summary</span>
            </div>

            <div className="p-3">
                {/* Main P&L - Inline */}
                <div className="flex items-center justify-between mb-3">
                    <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Today's P&L</span>
                    <div className="flex items-center gap-2">
                        <span className={`text-lg font-bold font-mono ${isProfit ? 'price-up' : 'price-down'}`}>
                            {isProfit ? '+' : ''}${Math.abs(netPnl).toFixed(2)}
                        </span>
                        {isProfit ? (
                            <TrendingUp className="w-4 h-4 text-green-500" />
                        ) : (
                            <TrendingDown className="w-4 h-4 text-red-500" />
                        )}
                    </div>
                </div>

                {/* Stats Row */}
                <div className="flex items-center gap-4 text-xs">
                    <div className="flex items-center gap-1">
                        <Trophy className="w-3 h-3" style={{ color: 'var(--accent-warning)' }} />
                        <span style={{ color: 'var(--text-muted)' }}>Win</span>
                        <span className="font-bold">{winRate.toFixed(0)}%</span>
                    </div>
                    <div className="flex items-center gap-1">
                        <Target className="w-3 h-3" style={{ color: 'var(--accent-primary)' }} />
                        <span style={{ color: 'var(--text-muted)' }}>Trades</span>
                        <span className="font-bold">{totalTrades}</span>
                    </div>
                    <span style={{ color: 'var(--text-muted)' }}>({openPositions} open)</span>
                </div>
            </div>
        </div>
    );
};
