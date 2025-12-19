import { Activity } from 'lucide-react';
import { useWebSocket } from './hooks/useWebSocket';

// Components
import { Header } from './components/Header';
import { CoinTabs } from './components/CoinTabs';
import { KPIStats } from './components/KPIStats';
import { ChartSection } from './components/ChartSection';
import { PositionsPanel } from './components/PositionsPanel';
import { TradeHistory } from './components/TradeHistory';

function App() {
    const { data, connected, sendMessage } = useWebSocket('ws://localhost:8000/ws');

    const handleCoinChange = (symbol) => {
        sendMessage({ action: 'set_active', symbol });
    };

    if (!data) return (
        <div className="min-h-screen flex items-center justify-center bg-bg-primary text-gray-500">
            <div className="flex flex-col items-center gap-4 animate-pulse">
                <Activity className="w-12 h-12 text-blue-500" />
                <p>Connecting to Trading Engine...</p>
            </div>
        </div>
    );

    const activeSymbol = data?.active_symbol || 'BTCUSDT';
    const activeCoinData = data?.coins?.[activeSymbol] || {};

    return (
        <div className="min-h-screen bg-bg-primary text-gray-100 p-6">
            <div className="max-w-[1600px] mx-auto space-y-6">

                {/* Header & Navigation */}
                <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                    <Header connected={connected} />
                    <CoinTabs
                        coins={Object.keys(data.coins)}
                        activeSymbol={activeSymbol}
                        onSelect={handleCoinChange}
                    />
                </div>

                {/* Global Stats */}
                <KPIStats data={data} activeCoinData={activeCoinData} />

                {/* Main Workspace */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

                    {/* Left Column: Charts & History */}
                    <div className="lg:col-span-2 space-y-6">
                        <ChartSection
                            activeSymbol={activeSymbol}
                            activeCoinData={activeCoinData}
                        />
                        <TradeHistory trades={data.trades} />
                    </div>

                    {/* Right Column: Positions */}
                    <PositionsPanel
                        activeCoinData={activeCoinData}
                        tradeStats={{
                            trades_in_window: data.trades_in_window,
                            total_fees: data.total_fees
                        }}
                    />
                </div>
            </div>
        </div>
    );
}

export default App;
