/**
 * PositionsPanel - Active positions display (Compact)
 */

import { Layers } from 'lucide-react';

export const PositionsPanel = ({ activeCoinData, tradeStats }) => {
    const formatCurrency = (val) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(val || 0);
    const positions = activeCoinData?.positions || [];

    return (
        <div className="glass-card h-full flex flex-col overflow-hidden">
            {/* Header */}
            <div className="flex justify-between items-center px-3 py-2 border-b" style={{ borderColor: 'var(--border-subtle)' }}>
                <h3 className="font-semibold text-xs flex items-center gap-2">
                    <Layers className="w-3.5 h-3.5" style={{ color: 'var(--accent-primary)' }} />
                    Active Positions
                </h3>
                <span className="badge badge-primary text-[10px]">{positions.length}</span>
            </div>

            {/* Positions List */}
            <div className="flex-1 overflow-y-auto p-2">
                {positions.length > 0 ? (
                    <div className="space-y-2">
                        {positions.map((pos) => (
                            <div key={pos.id} className="rounded-lg p-2 relative" style={{ background: 'var(--bg-tertiary)' }}>
                                <div className={`absolute top-0 left-0 w-1 h-full rounded-l ${pos.side === 'LONG' ? 'bg-green-500' : 'bg-red-500'}`} />

                                <div className="flex justify-between items-center pl-2">
                                    <div className="flex items-center gap-2">
                                        <span className={`text-[10px] font-bold ${pos.side === 'LONG' ? 'text-green-400' : 'text-red-400'}`}>
                                            {pos.side}
                                        </span>
                                        <span className="text-xs text-gray-400">x{pos.size.toFixed(4)}</span>
                                    </div>
                                    <span className={`font-mono font-bold text-xs ${pos.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                        {formatCurrency(pos.unrealized_pnl)}
                                    </span>
                                </div>

                                <div className="flex justify-between items-center pl-2 mt-1 text-[10px] text-gray-500">
                                    <span>Entry: ${pos.entry_price.toFixed(2)}</span>
                                    <span>
                                        <span className="text-green-500">{pos.take_profit.toFixed(0)}</span>
                                        <span className="mx-1">/</span>
                                        <span className="text-red-500">{pos.stop_loss.toFixed(0)}</span>
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>
                ) : (
                    <div className="h-full flex items-center justify-center text-xs text-gray-500">
                        No active positions
                    </div>
                )}
            </div>
        </div>
    );
};
