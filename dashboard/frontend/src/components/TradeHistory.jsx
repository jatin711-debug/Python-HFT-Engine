/**
 * TradeHistory - Market execution log (Compact)
 */

import { ArrowUpRight, ArrowDownRight, History } from 'lucide-react';

export const TradeHistory = ({ trades }) => {
    const formatCurrency = (val) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(val || 0);

    return (
        <div className="glass-card h-full flex flex-col overflow-hidden">
            {/* Header */}
            <div className="flex justify-between items-center px-3 py-2 border-b" style={{ borderColor: 'var(--border-subtle)' }}>
                <h3 className="font-semibold text-xs flex items-center gap-2">
                    <History className="w-3.5 h-3.5" style={{ color: 'var(--accent-warning)' }} />
                    Market Execution Log
                </h3>
                <span className="badge badge-primary text-[10px]">{trades.length}</span>
            </div>

            {/* Table */}
            <div className="flex-1 overflow-y-auto">
                <table className="w-full text-xs">
                    <thead className="sticky top-0" style={{ background: 'var(--bg-secondary)' }}>
                        <tr className="text-[10px] uppercase" style={{ color: 'var(--text-muted)' }}>
                            <th className="px-3 py-2 text-left">Time</th>
                            <th className="px-3 py-2 text-left">Symbol</th>
                            <th className="px-3 py-2 text-left">Side</th>
                            <th className="px-3 py-2 text-right">Net P&L</th>
                            <th className="px-3 py-2 text-right">Fee</th>
                        </tr>
                    </thead>
                    <tbody>
                        {trades.slice().reverse().slice(0, 10).map((trade) => (
                            <tr key={trade.id} className="border-t hover:bg-white/5" style={{ borderColor: 'var(--border-subtle)' }}>
                                <td className="px-3 py-2 font-mono" style={{ color: 'var(--text-muted)' }}>{trade.exit_time}</td>
                                <td className="px-3 py-2 font-medium">{trade.symbol}</td>
                                <td className="px-3 py-2">
                                    <span className={`flex items-center gap-1 ${trade.side === 'LONG' ? 'text-green-400' : 'text-red-400'}`}>
                                        {trade.side === 'LONG' ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
                                        {trade.side}
                                    </span>
                                </td>
                                <td className={`px-3 py-2 text-right font-mono font-medium ${trade.net_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                    {formatCurrency(trade.net_pnl)}
                                </td>
                                <td className="px-3 py-2 text-right font-mono" style={{ color: 'var(--text-muted)' }}>
                                    {formatCurrency(trade.fee)}
                                </td>
                            </tr>
                        ))}
                        {trades.length === 0 && (
                            <tr>
                                <td colSpan="5" className="px-3 py-6 text-center" style={{ color: 'var(--text-muted)' }}>
                                    No trades executed yet
                                </td>
                            </tr>
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
};
