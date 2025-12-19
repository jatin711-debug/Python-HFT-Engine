/**
 * Trading Slice - WebSocket data and trading state
 */

import { createSlice, createSelector } from '@reduxjs/toolkit';

const initialState = {
    // Connection
    connected: false,
    lastUpdate: null,

    // Global stats
    totalPnl: 0,
    totalFees: 0,
    netPnl: 0,
    winRate: 0,
    totalTrades: 0,
    openPositions: 0,

    // Rate limiting
    tradesInWindow: 0,
    canTrade: true,
    nextTradeIn: 0,

    // Active symbol
    activeSymbol: 'BTCUSDT',

    // Coins data - keyed by symbol
    coins: {},

    // Recent trades (all coins)
    trades: [],
};

const tradingSlice = createSlice({
    name: 'trading',
    initialState,
    reducers: {
        setConnected: (state, action) => {
            state.connected = action.payload;
        },

        // Main update from WebSocket
        updateFromWebSocket: (state, action) => {
            const data = action.payload;

            state.lastUpdate = data.timestamp;
            state.connected = data.connected;
            state.activeSymbol = data.active_symbol;

            // Global stats
            state.totalPnl = data.total_pnl;
            state.totalFees = data.total_fees;
            state.netPnl = data.net_pnl;
            state.winRate = data.win_rate;
            state.totalTrades = data.total_trades;
            state.openPositions = data.open_positions;

            // Rate limiting
            state.tradesInWindow = data.trades_in_window;
            state.canTrade = data.can_trade;
            state.nextTradeIn = data.next_trade_in;

            // Coins
            state.coins = data.coins;

            // Trades
            state.trades = data.trades;
        },

        setActiveSymbol: (state, action) => {
            state.activeSymbol = action.payload;
        },
    },
});

export const {
    setConnected,
    updateFromWebSocket,
    setActiveSymbol,
} = tradingSlice.actions;

// Base selectors (simple property access - no memoization needed)
export const selectConnected = (state) => state.trading.connected;
export const selectActiveSymbol = (state) => state.trading.activeSymbol;
export const selectCoins = (state) => state.trading.coins;
export const selectTrades = (state) => state.trading.trades;

// Memoized selectors (return derived/computed values)
export const selectActiveCoinData = createSelector(
    [selectActiveSymbol, selectCoins],
    (symbol, coins) => coins[symbol] || {}
);

export const selectGlobalStats = createSelector(
    [(state) => state.trading],
    (trading) => ({
        totalPnl: trading.totalPnl,
        totalFees: trading.totalFees,
        netPnl: trading.netPnl,
        winRate: trading.winRate,
        totalTrades: trading.totalTrades,
        openPositions: trading.openPositions,
        tradesInWindow: trading.tradesInWindow,
        canTrade: trading.canTrade,
        nextTradeIn: trading.nextTradeIn,
    })
);

export default tradingSlice.reducer;

