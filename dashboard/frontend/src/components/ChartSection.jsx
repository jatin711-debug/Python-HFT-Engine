import { useMemo } from 'react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { Card, Badge } from './ui';

export const ChartSection = ({ activeSymbol, activeCoinData }) => {
    const chartData = useMemo(() => {
        return activeCoinData.prices?.map((p, i) => ({
            time: activeCoinData.timestamps?.[i] || '',
            price: p
        })) || [];
    }, [activeCoinData.prices, activeCoinData.timestamps]);

    const formatPct = (val) => `${(val || 0) >= 0 ? '+' : ''}${(val || 0).toFixed(2)}%`;

    return (
        <Card className="h-[400px] flex flex-col">
            <div className="flex justify-between items-center mb-4">
                <div>
                    <h2 className="text-lg font-bold flex items-center gap-2">
                        {activeSymbol}
                        <span className={`text-sm font-normal ${activeCoinData.price_change_pct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                            {formatPct(activeCoinData.price_change_pct)}
                        </span>
                    </h2>
                    <p className="text-2xl font-mono font-bold">${activeCoinData.price?.toFixed(2)}</p>
                </div>
                <div className="flex gap-2">
                    <Badge type={activeCoinData.signal_direction === 'BUY' ? 'success' : activeCoinData.signal_direction === 'SELL' ? 'danger' : 'neutral'}>
                        SIGNAL: {activeCoinData.signal_direction} ({(activeCoinData.signal_confidence * 100).toFixed(0)}%)
                    </Badge>
                </div>
            </div>

            <div className="flex-1 w-full min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData}>
                        <defs>
                            <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <XAxis dataKey="time" hide />
                        <YAxis
                            domain={['auto', 'auto']}
                            orientation="right"
                            tick={{ fill: '#6b7280', fontSize: 11 }}
                            tickFormatter={(val) => val.toFixed(2)}
                        />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#1f2937', borderColor: '#374151', borderRadius: '8px' }}
                            formatter={(val) => [`$${val.toFixed(2)}`, 'Price']}
                        />
                        <Area
                            type="monotone"
                            dataKey="price"
                            stroke="#3b82f6"
                            strokeWidth={2}
                            fillOpacity={1}
                            fill="url(#colorPrice)"
                        />
                        {activeCoinData.positions?.map((pos) => (
                            <ReferenceLine
                                key={pos.id}
                                y={pos.entry_price}
                                stroke={pos.side === 'LONG' ? '#10b981' : '#ef4444'}
                                strokeDasharray="3 3"
                            />
                        ))}
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </Card>
    );
};
