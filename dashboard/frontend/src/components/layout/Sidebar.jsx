/**
 * Sidebar Component - Left icon navigation
 */

import { useState } from 'react';
import {
    LayoutDashboard,
    Eye,
    LineChart,
    History,
    BarChart3,
    Settings,
    Zap,
} from 'lucide-react';

const SIDEBAR_ITEMS = [
    { id: 'dashboard', icon: LayoutDashboard, label: 'Dashboard', active: true },
    { id: 'watchlist', icon: Eye, label: 'Watchlist' },
    { id: 'positions', icon: LineChart, label: 'Positions' },
    { id: 'history', icon: History, label: 'Trade History' },
    { id: 'analytics', icon: BarChart3, label: 'Analytics' },
];

export const Sidebar = ({ activeView, onViewChange }) => {
    const [hoveredItem, setHoveredItem] = useState(null);

    return (
        <aside className="sidebar">
            {/* Main navigation */}
            <div className="flex flex-col gap-2 flex-1">
                {SIDEBAR_ITEMS.map((item) => (
                    <button
                        key={item.id}
                        onClick={() => onViewChange?.(item.id)}
                        onMouseEnter={() => setHoveredItem(item.id)}
                        onMouseLeave={() => setHoveredItem(null)}
                        className={`sidebar-item ${activeView === item.id ? 'active' : ''}`}
                    >
                        <item.icon className="w-5 h-5" />
                        <span className="sidebar-tooltip">{item.label}</span>
                    </button>
                ))}
            </div>

            {/* Divider */}
            <div style={{
                width: '32px',
                height: '1px',
                background: 'var(--border-default)',
                margin: '8px 0'
            }} />

            {/* Bottom items */}
            <div className="flex flex-col gap-2">
                {/* Auto-trade indicator */}
                <button
                    className="sidebar-item"
                    onMouseEnter={() => setHoveredItem('auto-trade')}
                    onMouseLeave={() => setHoveredItem(null)}
                    style={{ color: 'var(--accent-success)' }}
                >
                    <Zap className="w-5 h-5" />
                    <span className="sidebar-tooltip">Auto-Trade Active</span>
                </button>

                {/* Settings */}
                <button
                    onClick={() => onViewChange?.('settings')}
                    onMouseEnter={() => setHoveredItem('settings')}
                    onMouseLeave={() => setHoveredItem(null)}
                    className={`sidebar-item ${activeView === 'settings' ? 'active' : ''}`}
                >
                    <Settings className="w-5 h-5" />
                    <span className="sidebar-tooltip">Settings</span>
                </button>
            </div>
        </aside>
    );
};
