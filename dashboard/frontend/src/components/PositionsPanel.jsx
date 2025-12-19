import { Layers, Zap, AlertCircle } from 'lucide-react';
import { Card, Badge } from './ui';

export const PositionsPanel = ({ activeCoinData, tradeStats }) => {
    const formatCurrency = (val) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(val || 0);

    return (
        <div className="space-y-6">
            {/* Active Positions */}
            <Card className="min-h-[300px]">
                <div className="flex justify-between items-center mb-4">
                    <h3 className="font-bold flex items-center gap-2">
                        <Layers className="w-4 h-4 text-blue-400" />
                        Active Positions
                    </h3>
                    <Badge>{activeCoinData.positions?.length || 0}</Badge>
                </div>

                <div className="space-y-3">
                    {activeCoinData.positions?.map((pos) => (
                        <div key={pos.id} className="bg-bg-primary rounded-lg p-3 border border-border relative overflow-hidden group">
                            <div className={`absolute top-0 left-0 w-1 h-full ${pos.side === 'LONG' ? 'bg-emerald-500' : 'bg-red-500'}`} />

                            <div className="flex justify-between items-start mb-2">
                                <div>
                                    <span className={`text-xs font-bold ${pos.side === 'LONG' ? 'text-emerald-400' : 'text-red-400'}`}>
                                        {pos.side}
                                    </span>
                                    <span className="text-xs text-gray-500 ml-2 font-mono">x{pos.size.toFixed(4)}</span>
                                </div>
                                <span className={`font-mono font-bold text-sm ${pos.unrealized_pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                    {formatCurrency(pos.unrealized_pnl)}
                                </span>
                            </div>

                            <div className="grid grid-cols-2 gap-2 text-xs text-gray-400">
                                <div>
                                    <div className="text-[10px] uppercase">Entry</div>
                                    <div>${pos.entry_price.toFixed(2)}</div>
                                </div>
                                <div className="text-right">
                                    <div className="text-[10px] uppercase">TP / SL</div>
                                    <div>
                                        <span className="text-emerald-500">{pos.take_profit.toFixed(0)}</span>
                                        <span className="text-gray-600 mx-1">/</span>
                                        <span className="text-red-500">{pos.stop_loss.toFixed(0)}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    ))}

                    {(!activeCoinData.positions || activeCoinData.positions.length === 0) && (
                        <div className="text-center py-12 text-gray-500 text-sm border-2 border-dashed border-gray-800 rounded-lg">
                            No active positions
                        </div>
                    )}
                </div>
            </Card>

            {/* System Status */}
            <Card>
                <h3 className="font-bold flex items-center gap-2 mb-4">
                    <Zap className="w-4 h-4 text-amber-400" />
                    System Status
                </h3>
                <div className="space-y-4 text-sm">
                    <div className="flex justify-between items-center">
                        <span className="text-gray-400">Rate Limit Window</span>
                        <span className="font-mono">60s</span>
                    </div>
                    <div className="flex justify-between items-center">
                        <span className="text-gray-400">Current Load</span>
                        <span className="font-mono text-blue-400">{tradeStats.trades_in_window} trades</span>
                    </div>
                    <div className="flex justify-between items-center">
                        <span className="text-gray-400">Est. Fees Paid</span>
                        <span className="font-mono text-red-400">{formatCurrency(tradeStats.total_fees)}</span>
                    </div>
                    <div className="mt-4 pt-4 border-t border-gray-700">
                        <div className="flex items-center gap-2 text-xs text-gray-500">
                            <AlertCircle className="w-3 h-3" />
                            <span>Max 5 new positions / minute</span>
                        </div>
                    </div>
                </div>
            </Card>
        </div>
    );
};
