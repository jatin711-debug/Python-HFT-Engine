import { useSelector } from 'react-redux';
import { Activity } from 'lucide-react';
import { useWebSocket } from './hooks/useWebSocket';

// Redux selectors
import {
    selectConnected,
    selectActiveSymbol,
    selectActiveCoinData,
    selectCoins,
    selectGlobalStats,
    selectTrades,
} from './store/tradingSlice';

// Components
import { Header } from './components/Header';
import { CoinTabs } from './components/CoinTabs';
import { KPIStats } from './components/KPIStats';
import { ChartSection } from './components/ChartSection';
import { PositionsPanel } from './components/PositionsPanel';
import { TradeHistory } from './components/TradeHistory';

function App() {
    // WebSocket connection (dispatches to Redux)
    const { sendMessage } = useWebSocket('ws://localhost:8000/ws');

    // Redux state
    const connected = useSelector(selectConnected);
    const activeSymbol = useSelector(selectActiveSymbol);
    const activeCoinData = useSelector(selectActiveCoinData);
    const coins = useSelector(selectCoins);
    const globalStats = useSelector(selectGlobalStats);
    const trades = useSelector(selectTrades);

    // Loading state
    if (!connected || Object.keys(coins).length === 0) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-bg-primary text-gray-500">
                <div className="flex flex-col items-center gap-4 animate-pulse">
                    <Activity className="w-12 h-12 text-blue-500" />
                    <p>Connecting to Trading Engine...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-bg-primary text-gray-100 p-6">
            <div className="max-w-[1600px] mx-auto space-y-6">
                {/* Header & Navigation */}
                <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                    <Header connected={connected} />
                    <CoinTabs
                        coins={Object.keys(coins)}
                        activeSymbol={activeSymbol}
                        sendMessage={sendMessage}
                        data={{ coins }}
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
