import { Wallet, TrendingUp, Layers, Clock, AlertCircle } from 'lucide-react';
import { Card, Badge } from './ui';

export const KPIStats = ({ data, activeCoinData }) => {
    const formatCurrency = (val) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(val || 0);

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Net P&L */}
            <Card>
                <div className="flex justify-between items-start mb-2">
                    <span className="text-gray-400 text-sm font-medium">Net P&L</span>
                    <Wallet className="w-4 h-4 text-gray-500" />
                </div>
                <div className={`text-2xl font-bold ${data.net_pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {formatCurrency(data.net_pnl)}
                </div>
                <div className="text-xs text-gray-500 mt-1 flex justify-between">
                    <span>Gross: {formatCurrency(data.total_pnl)}</span>
                    <span>Fees: <span className="text-red-400">-{formatCurrency(data.total_fees)}</span></span>
                </div>
            </Card>

            {/* Win Rate */}
            <Card>
                <div className="flex justify-between items-start mb-2">
                    <span className="text-gray-400 text-sm font-medium">Win Rate</span>
                    <TrendingUp className="w-4 h-4 text-gray-500" />
                </div>
                <div className="text-2xl font-bold text-white">
                    {data.win_rate.toFixed(1)}%
                </div>
                <div className="text-xs text-gray-500 mt-1">
                    {data.total_trades} Total Trades
                </div>
            </Card>

            {/* Active Positions */}
            <Card>
                <div className="flex justify-between items-start mb-2">
                    <span className="text-gray-400 text-sm font-medium">Active Positions</span>
                    <Layers className="w-4 h-4 text-gray-500" />
                </div>
                <div className="text-2xl font-bold text-blue-400">
                    {activeCoinData.position_count || 0}
                </div>
                <div className="flex gap-2 mt-1">
                    <Badge type="success">{activeCoinData.long_count || 0} LONG</Badge>
                    <Badge type="danger">{activeCoinData.short_count || 0} SHORT</Badge>
                </div>
            </Card>

            {/* Trade Velocity */}
            <Card>
                <div className="flex justify-between items-start mb-2">
                    <span className="text-gray-400 text-sm font-medium">Trade Velocity</span>
                    <Clock className="w-4 h-4 text-gray-500" />
                </div>
                <div className="flex items-center gap-2">
                    <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
                        <div
                            className={`h-full ${data.can_trade ? 'bg-blue-500' : 'bg-red-500'}`}
                            style={{ width: `${(data.trades_in_window / 5) * 100}%` }}
                        />
                    </div>
                    <span className="text-sm font-bold">{data.trades_in_window}/5</span>
                </div>
                <div className="text-xs text-gray-500 mt-2 flex justify-between items-center">
                    <span>{data.can_trade ? 'Trading Active' : `Cooldown: ${data.next_trade_in}s`}</span>
                    {!data.can_trade && <AlertCircle className="w-3 h-3 text-red-500" />}
                </div>
            </Card>
        </div>
    );
};
