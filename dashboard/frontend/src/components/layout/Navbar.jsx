/**
 * Navbar Component - Top navigation bar
 */

import { useSelector, useDispatch } from 'react-redux';
import {
    Activity,
    Search,
    Settings,
    Sun,
    Moon,
    Bitcoin,
    TrendingUp,
    Wifi,
    WifiOff,
} from 'lucide-react';
import { toggleTheme, selectTheme } from '../../store/themeSlice';
import { selectMarketType, selectConnected, selectGlobalStats, setMarketType } from '../../store/tradingSlice';

export const Navbar = () => {
    const dispatch = useDispatch();
    const theme = useSelector(selectTheme);
    const marketType = useSelector(selectMarketType);
    const connected = useSelector(selectConnected);
    const globalStats = useSelector(selectGlobalStats);

    const handleMarketSwitch = (type) => {
        if (type !== marketType) {
            dispatch(setMarketType(type));
        }
    };

    const handleThemeToggle = () => {
        dispatch(toggleTheme());
    };

    const netPnl = globalStats?.netPnl || 0;
    const isProfit = netPnl >= 0;

    return (
        <nav className="navbar">
            {/* Brand */}
            <div className="navbar-brand">
                <Activity className="w-7 h-7" style={{ color: '#3b82f6' }} />
                <span>TradeEngine</span>
            </div>

            {/* Center - Market Type Tabs */}
            <div className="navbar-center">
                <div className="tab-group">
                    <button
                        onClick={() => handleMarketSwitch('crypto')}
                        className={`tab crypto ${marketType === 'crypto' ? 'active' : ''}`}
                    >
                        <Bitcoin className="w-4 h-4" />
                        Crypto
                    </button>
                    <button
                        onClick={() => handleMarketSwitch('stocks')}
                        className={`tab stocks ${marketType === 'stocks' ? 'active' : ''}`}
                    >
                        <TrendingUp className="w-4 h-4" />
                        Stocks
                    </button>
                </div>
            </div>

            {/* Right side */}
            <div className="navbar-right">
                {/* Search (future) */}
                <button className="btn btn-icon btn-ghost" title="Search">
                    <Search className="w-4 h-4" />
                </button>

                {/* P&L Display */}
                <div className="flex items-center gap-3 px-4 py-2 rounded-lg"
                    style={{ background: 'var(--bg-tertiary)' }}>
                    <span className="text-sm" style={{ color: 'var(--text-muted)' }}>P&L</span>
                    <span className={`price-change font-mono font-bold ${isProfit ? 'price-up' : 'price-down'}`}>
                        {isProfit ? '+' : ''}{netPnl.toFixed(2)}
                    </span>
                </div>

                {/* Connection Status */}
                <div className="flex items-center gap-2">
                    {connected ? (
                        <Wifi className="w-4 h-4" style={{ color: 'var(--accent-success)' }} />
                    ) : (
                        <WifiOff className="w-4 h-4" style={{ color: 'var(--accent-danger)' }} />
                    )}
                </div>

                {/* Theme Toggle */}
                <button
                    onClick={handleThemeToggle}
                    className="btn btn-icon btn-ghost"
                    title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
                >
                    {theme === 'dark' ? (
                        <Sun className="w-4 h-4" />
                    ) : (
                        <Moon className="w-4 h-4" />
                    )}
                </button>

                {/* Settings */}
                <button className="btn btn-icon btn-ghost" title="Settings">
                    <Settings className="w-4 h-4" />
                </button>
            </div>
        </nav>
    );
};
