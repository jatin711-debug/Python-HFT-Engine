/**
 * AnalyticsWidget - Trading statistics (Compact)
 */

import { useSelector } from 'react-redux';
import { TrendingUp, TrendingDown, Zap } from 'lucide-react';
import { selectGlobalStats, selectTrades } from '../../store/tradingSlice';

export const AnalyticsWidget = () => {
    const stats = useSelector(selectGlobalStats);
    const trades = useSelector(selectTrades);

    const totalTrades = stats?.totalTrades || 0;
    const wins = trades?.filter(t => t.net_pnl > 0).length || 0;
    const losses = trades?.filter(t => t.net_pnl <= 0).length || 0;
    const totalFees = stats?.totalFees || 0;
    const canTrade = stats?.canTrade;

    const bestTrade = trades?.length > 0 ? Math.max(...trades.map(t => t.net_pnl || 0)) : 0;
    const worstTrade = trades?.length > 0 ? Math.min(...trades.map(t => t.net_pnl || 0)) : 0;

    return (
        <div className="glass-card">
            <div className="card-header py-2 px-3">
                <span className="text-xs font-semibold">Analytics</span>
            </div>

            <div className="p-3 space-y-3">
                {/* Win/Loss Bar */}
                <div>
                    <div className="flex justify-between text-[10px] mb-1" style={{ color: 'var(--text-muted)' }}>
                        <span>{wins}W / {losses}L</span>
                        <span>{totalTrades > 0 ? ((wins / totalTrades) * 100).toFixed(0) : 0}%</span>
                    </div>
                    <div className="flex h-2 rounded overflow-hidden" style={{ background: 'var(--bg-tertiary)' }}>
                        <div style={{ width: `${totalTrades > 0 ? (wins / totalTrades) * 100 : 50}%`, background: 'var(--accent-success)' }} />
                        <div style={{ width: `${totalTrades > 0 ? (losses / totalTrades) * 100 : 50}%`, background: 'var(--accent-danger)' }} />
                    </div>
                </div>

                {/* Stats Grid */}
                <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="p-2 rounded" style={{ background: 'var(--bg-tertiary)' }}>
                        <div className="flex items-center gap-1 mb-1">
                            <TrendingUp className="w-3 h-3 text-green-500" />
                            <span style={{ color: 'var(--text-muted)' }}>Best</span>
                        </div>
                        <span className="font-bold text-green-500">+${bestTrade.toFixed(2)}</span>
                    </div>
                    <div className="p-2 rounded" style={{ background: 'var(--bg-tertiary)' }}>
                        <div className="flex items-center gap-1 mb-1">
                            <TrendingDown className="w-3 h-3 text-red-500" />
                            <span style={{ color: 'var(--text-muted)' }}>Worst</span>
                        </div>
                        <span className="font-bold text-red-500">-${Math.abs(worstTrade).toFixed(2)}</span>
                    </div>
                </div>

                {/* Fees & Status */}
                <div className="flex items-center justify-between text-xs pt-2 border-t" style={{ borderColor: 'var(--border-subtle)' }}>
                    <div className="flex items-center gap-1">
                        <Zap className="w-3 h-3" style={{ color: 'var(--accent-warning)' }} />
                        <span style={{ color: 'var(--text-muted)' }}>Fees:</span>
                        <span className="font-mono">${totalFees.toFixed(2)}</span>
                    </div>
                    <span className={`badge text-[10px] ${canTrade ? 'badge-success' : 'badge-warning'}`}>
                        {canTrade ? 'Ready' : 'Cooldown'}
                    </span>
                </div>
            </div>
        </div>
    );
};
