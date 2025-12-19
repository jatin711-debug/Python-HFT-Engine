import { useState, useEffect, useRef } from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'

function App() {
    const [data, setData] = useState(null)
    const [connected, setConnected] = useState(false)
    const wsRef = useRef(null)

    useEffect(() => {
        // Connect to WebSocket
        const connect = () => {
            const ws = new WebSocket('ws://localhost:8000/ws')
            wsRef.current = ws

            ws.onopen = () => {
                console.log('Connected to server')
                setConnected(true)
            }

            ws.onmessage = (event) => {
                const newData = JSON.parse(event.data)
                setData(newData)
            }

            ws.onclose = () => {
                console.log('Disconnected')
                setConnected(false)
                // Reconnect after 3 seconds
                setTimeout(connect, 3000)
            }

            ws.onerror = (error) => {
                console.error('WebSocket error:', error)
            }
        }

        connect()

        return () => {
            if (wsRef.current) {
                wsRef.current.close()
            }
        }
    }, [])

    // Format currency
    const formatCurrency = (value, decimals = 2) => {
        if (value === undefined || value === null) return '$0.00'
        const prefix = value >= 0 ? '+$' : '-$'
        return `${value >= 0 ? '+' : ''}$${Math.abs(value).toFixed(decimals)}`
    }

    // Format percentage
    const formatPercent = (value) => {
        if (value === undefined || value === null) return '0.00%'
        return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`
    }

    // Prepare chart data
    const chartData = data?.prices?.map((price, i) => ({
        time: data.timestamps?.[i] || i,
        price: price,
    })) || []

    return (
        <div className="dashboard">
            {/* Header */}
            <header className="header">
                <h1>
                    🚀 Crypto Trading Dashboard
                    <span className="symbol-badge">{data?.symbol || 'BTCUSDT'}</span>
                </h1>

                <div className="price-display">
                    <div className="price-main">
                        ${data?.price?.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || '0.00'}
                    </div>
                    <div className={`price-change ${data?.price_change_pct >= 0 ? 'positive' : 'negative'}`}>
                        {formatPercent(data?.price_change_pct)}
                    </div>
                </div>

                <div className="connection-status">
                    <span className={`status-dot ${connected ? 'connected' : 'disconnected'}`}></span>
                    {connected ? 'Connected' : 'Disconnected'}
                </div>
            </header>

            {/* Stats Grid */}
            <div className="grid">
                {/* P&L Card */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">💰 Total P&L</span>
                    </div>
                    <div className={`card-value ${(data?.total_pnl || 0) >= 0 ? 'positive' : 'negative'}`}>
                        {formatCurrency(data?.total_pnl)}
                    </div>
                    <div className="card-subtitle">
                        {formatPercent(data?.total_pnl_pct)} • Win Rate: {(data?.win_rate || 0).toFixed(1)}%
                    </div>
                </div>

                {/* Position Card */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">📊 Position</span>
                    </div>
                    <div className={`card-value ${data?.position_side === 'LONG' ? 'positive' : data?.position_side === 'SHORT' ? 'negative' : ''}`}>
                        {data?.position_side || 'NONE'}
                    </div>
                    {data?.position_side !== 'NONE' && (
                        <div className="position-details">
                            <div className="position-item">
                                <span className="position-label">Entry</span>
                                <span className="position-value">${data?.position_entry?.toFixed(2)}</span>
                            </div>
                            <div className="position-item">
                                <span className="position-label">Unrealized</span>
                                <span className={`position-value ${(data?.position_pnl || 0) >= 0 ? 'positive' : 'negative'}`}>
                                    {formatCurrency(data?.position_pnl)}
                                </span>
                            </div>
                        </div>
                    )}
                </div>

                {/* Signal Card */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">🎯 Signal</span>
                    </div>
                    <div className={`signal-badge ${data?.signal_direction?.toLowerCase() || 'hold'}`}>
                        {data?.signal_direction || 'HOLD'}
                    </div>
                    <div className="confidence-bar">
                        <div
                            className="confidence-fill"
                            style={{ width: `${(data?.signal_confidence || 0) * 100}%` }}
                        ></div>
                    </div>
                    <div className="card-subtitle">
                        Confidence: {((data?.signal_confidence || 0) * 100).toFixed(0)}% •
                        Strength: {((data?.signal_strength || 0) * 100).toFixed(0)}%
                    </div>
                </div>

                {/* Trade Count Card */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">📈 Trades</span>
                    </div>
                    <div className="card-value">{data?.trade_count || 0}</div>
                    <div className="card-subtitle">
                        {data?.trades?.filter(t => t.pnl > 0).length || 0} wins •
                        {data?.trades?.filter(t => t.pnl <= 0).length || 0} losses
                    </div>
                </div>

                {/* Price Chart */}
                <div className="card chart-container">
                    <div className="card-header">
                        <span className="card-title">📉 Price Chart (Last 50)</span>
                    </div>
                    <div className="chart-wrapper">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={chartData}>
                                <XAxis
                                    dataKey="time"
                                    stroke="#6b7280"
                                    tick={{ fill: '#9ca3af', fontSize: 11 }}
                                />
                                <YAxis
                                    domain={['auto', 'auto']}
                                    stroke="#6b7280"
                                    tick={{ fill: '#9ca3af', fontSize: 11 }}
                                    tickFormatter={(v) => `$${v.toLocaleString()}`}
                                />
                                <Tooltip
                                    contentStyle={{
                                        background: '#1f2937',
                                        border: '1px solid #374151',
                                        borderRadius: '8px',
                                    }}
                                    formatter={(value) => [`$${value.toLocaleString()}`, 'Price']}
                                />
                                <Line
                                    type="monotone"
                                    dataKey="price"
                                    stroke="#3b82f6"
                                    strokeWidth={2}
                                    dot={false}
                                />
                                {data?.position_entry > 0 && (
                                    <ReferenceLine
                                        y={data.position_entry}
                                        stroke="#f59e0b"
                                        strokeDasharray="5 5"
                                        label={{ value: 'Entry', fill: '#f59e0b', fontSize: 11 }}
                                    />
                                )}
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Strategy Signals */}
                <div className="card">
                    <div className="card-header">
                        <span className="card-title">🧠 Strategy Signals</span>
                    </div>
                    <div className="strategy-signals">
                        <div className="strategy-row">
                            <span className="strategy-name">Momentum Burst</span>
                            <span className="strategy-value">{(data?.momentum_signal || 0).toFixed(2)}</span>
                        </div>
                        <div className="strategy-row">
                            <span className="strategy-name">Mean Reversion</span>
                            <span className="strategy-value">{(data?.mean_reversion_signal || 0).toFixed(2)}</span>
                        </div>
                        <div className="strategy-row">
                            <span className="strategy-name">Volatility Breakout</span>
                            <span className="strategy-value">{(data?.volatility_signal || 0).toFixed(2)}</span>
                        </div>
                        <div className="strategy-row">
                            <span className="strategy-name">Order Flow</span>
                            <span className="strategy-value">{(data?.order_flow_signal || 0).toFixed(2)}</span>
                        </div>
                    </div>
                </div>

                {/* Trade History */}
                <div className="card trades-container">
                    <div className="card-header">
                        <span className="card-title">📋 Recent Trades</span>
                    </div>
                    <table className="trades-table">
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>Side</th>
                                <th>Entry</th>
                                <th>Exit</th>
                                <th>P&L</th>
                                <th>Reason</th>
                            </tr>
                        </thead>
                        <tbody>
                            {(data?.trades || []).slice().reverse().map((trade, i) => (
                                <tr key={i}>
                                    <td>{trade.exit_time}</td>
                                    <td>
                                        <span className={`trade-side ${trade.side?.toLowerCase()}`}>
                                            {trade.side}
                                        </span>
                                    </td>
                                    <td>${trade.entry_price?.toFixed(2)}</td>
                                    <td>${trade.exit_price?.toFixed(2)}</td>
                                    <td className={trade.pnl >= 0 ? 'positive' : 'negative'}>
                                        {formatCurrency(trade.pnl)}
                                    </td>
                                    <td>{trade.reason}</td>
                                </tr>
                            ))}
                            {(!data?.trades || data.trades.length === 0) && (
                                <tr>
                                    <td colSpan="6" style={{ textAlign: 'center', color: 'var(--text-secondary)' }}>
                                        No trades yet. Waiting for signals...
                                    </td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    )
}

export default App
