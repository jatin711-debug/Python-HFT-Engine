import { ArrowUpRight, ArrowDownRight } from 'lucide-react';
import { Card } from './ui';

export const TradeHistory = ({ trades }) => {
    const formatCurrency = (val) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(val || 0);

    return (
        <Card>
            <h3 className="text-sm font-medium text-gray-400 mb-4 uppercase tracking-wider">Market Execution Log</h3>
            <div className="overflow-x-auto">
                <table className="w-full text-sm text-left">
                    <thead className="text-xs text-gray-500 uppercase border-b border-gray-700">
                        <tr>
                            <th className="px-4 py-3">Time</th>
                            <th className="px-4 py-3">Symbol</th>
                            <th className="px-4 py-3">Side</th>
                            <th className="px-4 py-3 text-right">Net P&L</th>
                            <th className="px-4 py-3 text-right">Fee</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-800">
                        {trades.slice().reverse().map((trade) => (
                            <tr key={trade.id} className="hover:bg-white/5 transition-colors">
                                <td className="px-4 py-3 text-gray-400 font-mono text-xs">{trade.exit_time}</td>
                                <td className="px-4 py-3 font-medium">{trade.symbol}</td>
                                <td className="px-4 py-3">
                                    <span className={`inline-flex items-center gap-1 ${trade.side === 'LONG' ? 'text-emerald-400' : 'text-red-400'}`}>
                                        {trade.side === 'LONG' ? <ArrowUpRight className="w-3 h-3" /> : <ArrowDownRight className="w-3 h-3" />}
                                        {trade.side}
                                    </span>
                                </td>
                                <td className={`px-4 py-3 text-right font-medium ${trade.net_pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                    {formatCurrency(trade.net_pnl)}
                                </td>
                                <td className="px-4 py-3 text-right text-gray-500 text-xs">
                                    {formatCurrency(trade.fee)}
                                </td>
                            </tr>
                        ))}
                        {trades.length === 0 && (
                            <tr>
                                <td colSpan="5" className="px-4 py-8 text-center text-gray-500 italic">No trades executed yet</td>
                            </tr>
                        )}
                    </tbody>
                </table>
            </div>
        </Card>
    );
};
