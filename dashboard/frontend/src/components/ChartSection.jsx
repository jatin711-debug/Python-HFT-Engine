/**
 * Professional Chart Section using TradingView Lightweight Charts
 * 
 * Features:
 * - Real-time candlestick updates
 * - Multi-timeframe support (view only)
 * - Technical indicator overlays
 * - Position entry lines
 * - Volume histogram
 */

import { useEffect, useRef, useMemo, useCallback } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { createChart, ColorType, CrosshairMode, CandlestickSeries, HistogramSeries, LineSeries } from 'lightweight-charts';
import { Card, Badge } from './ui';
import { TrendingUp, TrendingDown, BarChart3 } from 'lucide-react';
import { TimeframeSelector } from './TimeframeSelector';
import { selectActiveCoinData, selectActiveSymbol } from '../store/tradingSlice';
import {
    selectViewTimeframe,
    selectChartType,
    selectSelectedIndicator,
    setChartType,
    setSelectedIndicator,
} from '../store/uiSlice';

// Aggregate 1s candles into higher timeframes
const aggregateCandles = (candles, targetTimeframe) => {
    if (!candles || candles.length === 0) return [];
    if (targetTimeframe === '1s') return candles;

    const intervalSeconds = {
        '5s': 5,
        '15s': 15,
        '1m': 60,
        '5m': 300,
        '15m': 900,
    };

    const interval = intervalSeconds[targetTimeframe] || 1;
    const aggregated = [];
    let bucket = null;
    let bucketStart = null;

    for (const candle of candles) {
        // Parse time (HH:MM:SS format)
        const timeParts = candle.time.split(':');
        const seconds = parseInt(timeParts[0]) * 3600 + parseInt(timeParts[1]) * 60 + parseInt(timeParts[2]);
        const bucketTime = Math.floor(seconds / interval) * interval;

        if (bucketStart !== bucketTime) {
            if (bucket) {
                aggregated.push(bucket);
            }
            bucketStart = bucketTime;
            bucket = {
                time: candle.time,
                open: candle.open,
                high: candle.high,
                low: candle.low,
                close: candle.close,
                volume: candle.volume,
            };
        } else if (bucket) {
            bucket.high = Math.max(bucket.high, candle.high);
            bucket.low = Math.min(bucket.low, candle.low);
            bucket.close = candle.close;
            bucket.volume += candle.volume;
        }
    }

    if (bucket) {
        aggregated.push(bucket);
    }

    return aggregated;
};

// Convert time string to Unix timestamp for chart (with error handling)
const timeToTimestamp = (timeStr, baseDate = new Date()) => {
    try {
        if (!timeStr || typeof timeStr !== 'string') {
            return Math.floor(Date.now() / 1000);
        }
        const parts = timeStr.split(':');
        if (parts.length < 3) {
            return Math.floor(Date.now() / 1000);
        }
        const [hours, minutes, seconds] = parts.map(Number);
        if (isNaN(hours) || isNaN(minutes) || isNaN(seconds)) {
            return Math.floor(Date.now() / 1000);
        }
        const date = new Date(baseDate);
        date.setHours(hours, minutes, seconds, 0);
        return Math.floor(date.getTime() / 1000);
    } catch (e) {
        console.warn('timeToTimestamp error:', e);
        return Math.floor(Date.now() / 1000);
    }
};

export const ChartSection = ({ sendMessage }) => {
    const dispatch = useDispatch();
    const chartContainerRef = useRef(null);
    const chartRef = useRef(null);
    const candleSeriesRef = useRef(null);
    const volumeSeriesRef = useRef(null);
    const indicatorSeriesRef = useRef({});
    const positionLinesRef = useRef([]);

    // Redux selectors
    const activeSymbol = useSelector(selectActiveSymbol);
    const activeCoinData = useSelector(selectActiveCoinData);
    const viewTimeframe = useSelector(selectViewTimeframe);
    const chartType = useSelector(selectChartType);
    const selectedIndicator = useSelector(selectSelectedIndicator);

    // Aggregate candles based on selected timeframe
    const chartCandles = useMemo(() => {
        const rawCandles = activeCoinData.candles || [];
        return aggregateCandles(rawCandles, viewTimeframe);
    }, [activeCoinData.candles, viewTimeframe]);

    // Format percentage
    const formatPct = (val) => `${(val || 0) >= 0 ? '+' : ''}${(val || 0).toFixed(2)}%`;

    // Initialize chart
    useEffect(() => {
        if (!chartContainerRef.current) return;

        const chart = createChart(chartContainerRef.current, {
            layout: {
                background: { type: ColorType.Solid, color: '#0d1117' },
                textColor: '#9ca3af',
            },
            grid: {
                vertLines: { color: '#1f2937' },
                horzLines: { color: '#1f2937' },
            },
            crosshair: {
                mode: CrosshairMode.Normal,
                vertLine: {
                    color: '#6b7280',
                    width: 1,
                    style: 2,
                },
                horzLine: {
                    color: '#6b7280',
                    width: 1,
                    style: 2,
                },
            },
            rightPriceScale: {
                borderColor: '#374151',
                scaleMargins: {
                    top: 0.1,
                    bottom: 0.2,
                },
            },
            timeScale: {
                borderColor: '#374151',
                timeVisible: true,
                secondsVisible: true,
            },
            handleScroll: {
                vertTouchDrag: false,
            },
        });

        // Create candlestick series (v4.x API)
        const candleSeries = chart.addSeries(CandlestickSeries, {
            upColor: '#10b981',
            downColor: '#ef4444',
            borderUpColor: '#10b981',
            borderDownColor: '#ef4444',
            wickUpColor: '#10b981',
            wickDownColor: '#ef4444',
        });

        // Create volume series (v4.x API)
        const volumeSeries = chart.addSeries(HistogramSeries, {
            priceFormat: {
                type: 'volume',
            },
            priceScaleId: 'volume',
        });

        // Configure volume scale
        chart.priceScale('volume').applyOptions({
            scaleMargins: {
                top: 0.85,
                bottom: 0,
            },
        });

        chartRef.current = chart;
        candleSeriesRef.current = candleSeries;
        volumeSeriesRef.current = volumeSeries;

        // Handle resize
        const handleResize = () => {
            if (chartContainerRef.current && chart) {
                chart.applyOptions({
                    width: chartContainerRef.current.clientWidth,
                    height: chartContainerRef.current.clientHeight,
                });
            }
        };

        window.addEventListener('resize', handleResize);
        handleResize();

        return () => {
            window.removeEventListener('resize', handleResize);
            chart.remove();
            chartRef.current = null;
            candleSeriesRef.current = null;
            volumeSeriesRef.current = null;
        };
    }, []);

    // Update chart data
    useEffect(() => {
        try {
            if (!candleSeriesRef.current || !volumeSeriesRef.current) return;
            if (!chartCandles || chartCandles.length === 0) return;

            const baseDate = new Date();
            baseDate.setHours(0, 0, 0, 0);

            // Convert candles to chart format
            const candleData = chartCandles.map((c, i) => ({
                time: timeToTimestamp(c?.time, baseDate) + i, // Add index to ensure unique times
                open: c?.open || 0,
                high: c?.high || 0,
                low: c?.low || 0,
                close: c?.close || 0,
            }));

            const volumeData = chartCandles.map((c, i) => ({
                time: timeToTimestamp(c?.time, baseDate) + i,
                value: c?.volume || 0,
                color: (c?.close || 0) >= (c?.open || 0) ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)',
            }));

            candleSeriesRef.current.setData(candleData);
            volumeSeriesRef.current.setData(volumeData);

            // Fit content
            chartRef.current?.timeScale().fitContent();
        } catch (e) {
            console.error('Chart data update error:', e);
        }
    }, [chartCandles]);

    // Update indicator overlays
    useEffect(() => {
        if (!chartRef.current) return;

        const chart = chartRef.current;
        const indicators = activeCoinData.indicators || {};

        // Remove existing indicator series
        Object.values(indicatorSeriesRef.current).forEach((series) => {
            try {
                chart.removeSeries(series);
            } catch (e) {
                // Series might already be removed
            }
        });
        indicatorSeriesRef.current = {};

        if (selectedIndicator === 'none' || chartCandles.length === 0) return;

        const baseDate = new Date();
        baseDate.setHours(0, 0, 0, 0);

        // Create indicator data (use last indicator value for all points for now)
        const createLineData = (value, color) => {
            if (!value) return [];
            return chartCandles.map((c, i) => ({
                time: timeToTimestamp(c.time, baseDate) + i,
                value: value,
            }));
        };

        if (selectedIndicator === 'bb' && indicators.bb_middle) {
            const bbUpper = chart.addSeries(LineSeries, {
                color: '#10b981',
                lineWidth: 1,
                lineStyle: 2,
                priceLineVisible: false,
            });
            const bbMiddle = chart.addSeries(LineSeries, {
                color: '#6b7280',
                lineWidth: 1,
                priceLineVisible: false,
            });
            const bbLower = chart.addSeries(LineSeries, {
                color: '#ef4444',
                lineWidth: 1,
                lineStyle: 2,
                priceLineVisible: false,
            });

            bbUpper.setData(createLineData(indicators.bb_upper));
            bbMiddle.setData(createLineData(indicators.bb_middle));
            bbLower.setData(createLineData(indicators.bb_lower));

            indicatorSeriesRef.current = { bbUpper, bbMiddle, bbLower };
        }

        if (selectedIndicator === 'ema' && indicators.ema_fast) {
            const emaFast = chart.addSeries(LineSeries, {
                color: '#3b82f6',
                lineWidth: 2,
                priceLineVisible: false,
            });
            const emaSlow = chart.addSeries(LineSeries, {
                color: '#8b5cf6',
                lineWidth: 2,
                priceLineVisible: false,
            });

            emaFast.setData(createLineData(indicators.ema_fast));
            emaSlow.setData(createLineData(indicators.ema_slow));

            indicatorSeriesRef.current = { emaFast, emaSlow };
        }
    }, [selectedIndicator, activeCoinData.indicators, chartCandles]);

    // Update position lines
    useEffect(() => {
        if (!candleSeriesRef.current) return;

        // Remove existing price lines
        positionLinesRef.current.forEach((line) => {
            try {
                candleSeriesRef.current.removePriceLine(line);
            } catch (e) {
                // Line might already be removed
            }
        });
        positionLinesRef.current = [];

        // Add position entry lines
        const positions = activeCoinData?.positions || [];
        positions.forEach((pos) => {
            const priceLine = candleSeriesRef.current.createPriceLine({
                price: pos.entry_price,
                color: pos.side === 'LONG' ? '#10b981' : '#ef4444',
                lineWidth: 2,
                lineStyle: 2,
                axisLabelVisible: true,
                title: `${pos.side} ${pos.id}`,
            });
            positionLinesRef.current.push(priceLine);
        });
    }, [activeCoinData?.positions]);

    return (
        <Card className="h-[450px] flex flex-col">
            {/* Header */}
            <div className="flex justify-between items-center mb-4">
                <div>
                    <h2 className="text-lg font-bold flex items-center gap-2">
                        {activeSymbol}
                        <span
                            className={`text-sm font-normal flex items-center gap-1 ${(activeCoinData?.price_change_pct || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'
                                }`}
                        >
                            {(activeCoinData?.price_change_pct || 0) >= 0 ? (
                                <TrendingUp className="w-4 h-4" />
                            ) : (
                                <TrendingDown className="w-4 h-4" />
                            )}
                            {formatPct(activeCoinData?.price_change_pct)}
                        </span>
                    </h2>
                    <p className="text-2xl font-mono font-bold">${activeCoinData?.price?.toFixed(2) || '0.00'}</p>
                </div>

                <div className="flex gap-3 items-center flex-wrap justify-end">
                    {/* Timeframe Selector */}
                    <TimeframeSelector />

                    {/* Chart Type Toggle */}
                    <div className="flex gap-1 bg-bg-secondary p-1 rounded">
                        <button
                            onClick={() => dispatch(setChartType('candlestick'))}
                            className={`px-2 py-1 text-xs rounded ${chartType === 'candlestick' ? 'bg-bg-card text-white' : 'text-gray-400'
                                }`}
                        >
                            <BarChart3 className="w-3 h-3" />
                        </button>
                        <button
                            onClick={() => dispatch(setChartType('line'))}
                            className={`px-2 py-1 text-xs rounded ${chartType === 'line' ? 'bg-bg-card text-white' : 'text-gray-400'
                                }`}
                        >
                            Line
                        </button>
                    </div>

                    {/* Indicator Selector */}
                    <select
                        value={selectedIndicator}
                        onChange={(e) => dispatch(setSelectedIndicator(e.target.value))}
                        className="bg-bg-secondary text-xs px-2 py-1 rounded border border-border text-gray-300"
                    >
                        <option value="none">No Indicator</option>
                        <option value="bb">Bollinger Bands</option>
                        <option value="ema">EMA (Fast/Slow)</option>
                        <option value="rsi">RSI</option>
                        <option value="macd">MACD</option>
                    </select>

                    {/* Signal Badge */}
                    <Badge
                        type={
                            activeCoinData?.signal_direction === 'BUY'
                                ? 'success'
                                : activeCoinData?.signal_direction === 'SELL'
                                    ? 'danger'
                                    : 'neutral'
                        }
                    >
                        {activeCoinData?.signal_direction || 'HOLD'} ({((activeCoinData?.signal_confidence || 0) * 100).toFixed(0)}%)
                    </Badge>
                </div>
            </div>

            {/* Indicator Values Display */}
            {selectedIndicator !== 'none' && (
                <div className="flex gap-3 mb-2 text-xs text-gray-400">
                    {selectedIndicator === 'rsi' && activeCoinData.indicators?.rsi && (
                        <span>
                            RSI: <span className="text-white font-mono">{activeCoinData.indicators.rsi.toFixed(2)}</span>
                        </span>
                    )}
                    {selectedIndicator === 'macd' && activeCoinData.indicators?.macd && (
                        <>
                            <span>
                                MACD:{' '}
                                <span className="text-white font-mono">{activeCoinData.indicators.macd.toFixed(4)}</span>
                            </span>
                            <span>
                                Signal:{' '}
                                <span className="text-white font-mono">
                                    {activeCoinData.indicators.macd_signal?.toFixed(4)}
                                </span>
                            </span>
                        </>
                    )}
                    {selectedIndicator === 'bb' && activeCoinData.indicators?.bb_middle && (
                        <span>
                            BB:{' '}
                            <span className="text-emerald-400 font-mono">
                                {activeCoinData.indicators.bb_upper?.toFixed(2)}
                            </span>{' '}
                            /{' '}
                            <span className="text-white font-mono">
                                {activeCoinData.indicators.bb_middle.toFixed(2)}
                            </span>{' '}
                            /{' '}
                            <span className="text-red-400 font-mono">
                                {activeCoinData.indicators.bb_lower?.toFixed(2)}
                            </span>
                        </span>
                    )}
                    {selectedIndicator === 'ema' && activeCoinData.indicators?.ema_fast && (
                        <>
                            <span>
                                EMA Fast:{' '}
                                <span className="text-blue-400 font-mono">
                                    {activeCoinData.indicators.ema_fast.toFixed(2)}
                                </span>
                            </span>
                            <span>
                                EMA Slow:{' '}
                                <span className="text-purple-400 font-mono">
                                    {activeCoinData.indicators.ema_slow?.toFixed(2)}
                                </span>
                            </span>
                        </>
                    )}
                </div>
            )}

            {/* Chart Container */}
            <div ref={chartContainerRef} className="flex-1 w-full min-h-0" />
        </Card>
    );
};
