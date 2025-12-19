import { useSelector, useDispatch } from 'react-redux';
import { Activity, TrendingUp, Bitcoin } from 'lucide-react';
import { useWebSocket } from './hooks/useWebSocket';

// Redux selectors and actions
import {
    selectConnected,
    selectActiveSymbol,
    selectActiveCoinData,
    selectCoins,
    selectGlobalStats,
    selectTrades,
    selectMarketType,
    selectMarketOpen,
    setMarketType,
} from './store/tradingSlice';

// Components
import { Header } from './components/Header';
import { CoinTabs } from './components/CoinTabs';
import { KPIStats } from './components/KPIStats';
import { ChartSection } from './components/ChartSection';
import { PositionsPanel } from './components/PositionsPanel';
import { TradeHistory } from './components/TradeHistory';

// WebSocket URLs
const WS_URLS = {
    crypto: 'ws://localhost:8000/ws',
    stocks: 'ws://localhost:8001/ws',
};

function App() {
    const dispatch = useDispatch();

    // Redux state
    const marketType = useSelector(selectMarketType);
    const marketOpen = useSelector(selectMarketOpen);
    const connected = useSelector(selectConnected);
    const activeSymbol = useSelector(selectActiveSymbol);
    const activeCoinData = useSelector(selectActiveCoinData);
    const coins = useSelector(selectCoins);
    const globalStats = useSelector(selectGlobalStats);
    const trades = useSelector(selectTrades);

    // WebSocket connection (switches based on market type)
    const wsUrl = WS_URLS[marketType];
    const { sendMessage } = useWebSocket(wsUrl);

    // Handle market type switch
    const handleMarketSwitch = (newMarketType) => {
        if (newMarketType !== marketType) {
            dispatch(setMarketType(newMarketType));
        }
    };

    // Loading state
    if (!connected || Object.keys(coins).length === 0) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-bg-primary text-gray-500">
                <div className="flex flex-col items-center gap-4 animate-pulse">
                    <Activity className="w-12 h-12 text-blue-500" />
                    <p>Connecting to {marketType === 'stocks' ? 'Stock' : 'Crypto'} Trading Engine...</p>

                    {/* Market Type Toggle - Always visible during loading */}
                    <div className="flex gap-2 mt-4">
                        <button
                            onClick={() => handleMarketSwitch('crypto')}
                            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${marketType === 'crypto'
                                    ? 'bg-orange-500/20 border border-orange-500 text-orange-400'
                                    : 'bg-bg-secondary border border-gray-700 text-gray-400 hover:border-gray-500'
                                }`}
                        >
                            <Bitcoin className="w-4 h-4" />
                            Crypto
                        </button>
                        <button
                            onClick={() => handleMarketSwitch('stocks')}
                            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${marketType === 'stocks'
                                    ? 'bg-green-500/20 border border-green-500 text-green-400'
                                    : 'bg-bg-secondary border border-gray-700 text-gray-400 hover:border-gray-500'
                                }`}
                        >
                            <TrendingUp className="w-4 h-4" />
                            Stocks
                        </button>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-bg-primary text-gray-100 p-6">
            <div className="max-w-[1600px] mx-auto space-y-6">
                {/* Market Type Tabs + Header + Navigation */}
                <div className="flex flex-col gap-4">
                    {/* Market Type Toggle */}
                    <div className="flex items-center justify-between">
                        <div className="flex gap-2">
                            <button
                                onClick={() => handleMarketSwitch('crypto')}
                                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all font-medium ${marketType === 'crypto'
                                        ? 'bg-orange-500/20 border-2 border-orange-500 text-orange-400'
                                        : 'bg-bg-secondary border border-gray-700 text-gray-400 hover:border-gray-500'
                                    }`}
                            >
                                <Bitcoin className="w-5 h-5" />
                                Crypto
                            </button>
                            <button
                                onClick={() => handleMarketSwitch('stocks')}
                                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all font-medium ${marketType === 'stocks'
                                        ? 'bg-green-500/20 border-2 border-green-500 text-green-400'
                                        : 'bg-bg-secondary border border-gray-700 text-gray-400 hover:border-gray-500'
                                    }`}
                            >
                                <TrendingUp className="w-5 h-5" />
                                Stocks
                                {marketType === 'stocks' && !marketOpen && (
                                    <span className="text-xs bg-red-500/20 text-red-400 px-2 py-0.5 rounded">
                                        Closed
                                    </span>
                                )}
                            </button>
                        </div>
                        <Header connected={connected} />
                    </div>

                    {/* Asset Tabs */}
                    <CoinTabs
                        coins={Object.keys(coins)}
                        activeSymbol={activeSymbol}
                        sendMessage={sendMessage}
                        data={{ coins }}
                        marketType={marketType}
                    />
                </div>

                {/* Global Stats */}
                <KPIStats
                    data={{
                        net_pnl: globalStats.netPnl,
                        total_pnl: globalStats.totalPnl,
                        total_fees: globalStats.totalFees,
                        win_rate: globalStats.winRate,
                        total_trades: globalStats.totalTrades,
                        trades_in_window: globalStats.tradesInWindow,
                        can_trade: globalStats.canTrade,
                        next_trade_in: globalStats.nextTradeIn,
                    }}
                    activeCoinData={activeCoinData}
                />

                {/* Main Workspace */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Left Column: Charts & History */}
                    <div className="lg:col-span-2 space-y-6">
                        <ChartSection sendMessage={sendMessage} />
                        <TradeHistory trades={trades} />
                    </div>

                    {/* Right Column: Positions */}
                    <PositionsPanel
                        activeCoinData={activeCoinData}
                        tradeStats={{
                            trades_in_window: globalStats.tradesInWindow,
                            total_fees: globalStats.totalFees,
                        }}
                    />
                </div>
            </div>
        </div>
    );
}

export default App;

