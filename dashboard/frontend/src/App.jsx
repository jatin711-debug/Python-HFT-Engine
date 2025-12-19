/**
 * TradeEngine Dashboard - Main Application
 * 
 * Professional trading terminal with:
 * - Left sidebar navigation
 * - Top navbar with market tabs
 * - Main chart area
 * - Right sidebar with widgets
 * - Dark/Light theme support
 */

import { useState, useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { Activity } from 'lucide-react';
import { useWebSocket } from './hooks/useWebSocket';

// Redux
import {
    selectConnected,
    selectActiveSymbol,
    selectActiveCoinData,
    selectCoins,
    selectGlobalStats,
    selectTrades,
    selectMarketType,
} from './store/tradingSlice';
import { selectTheme, setTheme } from './store/themeSlice';

// Layout Components
import { Navbar } from './components/layout/Navbar';
import { Sidebar } from './components/layout/Sidebar';

// Trading Components
import { ChartSection } from './components/ChartSection';
import { PositionsPanel } from './components/PositionsPanel';
import { TradeHistory } from './components/TradeHistory';
import { CoinTabs } from './components/CoinTabs';

// Widget Components
import { AccountCard } from './components/widgets/AccountCard';
import { SignalPanel } from './components/widgets/SignalPanel';
import { QuickTrade } from './components/widgets/QuickTrade';
import { WatchlistWidget } from './components/widgets/WatchlistWidget';
import { AnalyticsWidget } from './components/widgets/AnalyticsWidget';

// WebSocket URLs
const WS_URLS = {
    crypto: 'ws://localhost:8000/ws',
    stocks: 'ws://localhost:8001/ws',
};

function App() {
    const dispatch = useDispatch();

    // Theme
    const theme = useSelector(selectTheme);

    // Apply theme on mount and changes
    useEffect(() => {
        document.documentElement.setAttribute('data-theme', theme);
    }, [theme]);

    // Active view for sidebar
    const [activeView, setActiveView] = useState('dashboard');

    // Redux state
    const marketType = useSelector(selectMarketType);
    const connected = useSelector(selectConnected);
    const activeSymbol = useSelector(selectActiveSymbol);
    const activeCoinData = useSelector(selectActiveCoinData);
    const coins = useSelector(selectCoins);
    const globalStats = useSelector(selectGlobalStats);
    const trades = useSelector(selectTrades);

    // WebSocket
    const wsUrl = WS_URLS[marketType];
    const { sendMessage } = useWebSocket(wsUrl);

    // Loading state
    if (!connected || Object.keys(coins).length === 0) {
        return (
            <div className="min-h-screen flex items-center justify-center"
                style={{ background: 'var(--bg-primary)' }}>
                <div className="flex flex-col items-center gap-4">
                    <div className="relative">
                        <Activity className="w-16 h-16" style={{ color: 'var(--accent-primary)' }} />
                        <div className="absolute inset-0 animate-ping">
                            <Activity className="w-16 h-16 opacity-30" style={{ color: 'var(--accent-primary)' }} />
                        </div>
                    </div>
                    <div className="text-center">
                        <h2 className="text-xl font-bold mb-2">TradeEngine</h2>
                        <p style={{ color: 'var(--text-muted)' }}>
                            Connecting to {marketType === 'stocks' ? 'Stock' : 'Crypto'} Trading Engine...
                        </p>
                    </div>

                    {/* Theme toggle even during loading */}
                    <button
                        onClick={() => dispatch(setTheme(theme === 'dark' ? 'light' : 'dark'))}
                        className="btn btn-ghost mt-4"
                    >
                        Switch to {theme === 'dark' ? 'Light' : 'Dark'} Mode
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="dashboard-grid">
            {/* Top Navbar */}
            <Navbar />

            {/* Left Sidebar */}
            <Sidebar activeView={activeView} onViewChange={setActiveView} />

            {/* Main Content Area */}
            <main className="overflow-hidden" style={{
                background: 'var(--bg-primary)',
                padding: '12px',
            }}>
                <div style={{ height: '100%', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                    {/* Asset Tabs - Fixed height */}
                    <div style={{ flex: '0 0 auto' }}>
                        <CoinTabs
                            coins={Object.keys(coins)}
                            activeSymbol={activeSymbol}
                            sendMessage={sendMessage}
                            data={{ coins }}
                            marketType={marketType}
                        />
                    </div>

                    {/* Chart - Takes remaining space minus bottom panels */}
                    <div style={{ flex: '1 1 auto', minHeight: '250px', overflow: 'hidden' }}>
                        <ChartSection sendMessage={sendMessage} />
                    </div>

                    {/* Positions & History - Fixed 240px height */}
                    <div style={{
                        flex: '0 0 240px',
                        display: 'grid',
                        gridTemplateColumns: '1fr 1fr',
                        gap: '12px',
                    }}>
                        <PositionsPanel
                            activeCoinData={activeCoinData}
                            tradeStats={{
                                trades_in_window: globalStats.tradesInWindow,
                                total_fees: globalStats.totalFees,
                            }}
                        />
                        <TradeHistory trades={trades} />
                    </div>
                </div>
            </main>

            {/* Right Sidebar - Widgets */}
            <aside className="right-sidebar overflow-y-auto" style={{
                background: 'var(--bg-secondary)',
                borderLeft: '1px solid var(--border-default)',
                padding: '12px',
            }}>
                <div className="space-y-3">
                    <AccountCard />
                    <SignalPanel />
                    <QuickTrade sendMessage={sendMessage} />
                    <WatchlistWidget sendMessage={sendMessage} />
                    <AnalyticsWidget />
                </div>
            </aside>
        </div>
    );
}

export default App;
