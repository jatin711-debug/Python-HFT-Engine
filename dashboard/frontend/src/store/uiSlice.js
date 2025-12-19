/**
 * UI Slice - User interface state
 */

import { createSlice } from '@reduxjs/toolkit';

const TIMEFRAMES = ['1s', '5s', '15s', '1m', '5m', '15m'];

const initialState = {
    // Chart settings
    viewTimeframe: '1s',  // View-only, doesn't affect trading
    chartType: 'candlestick',  // 'candlestick' or 'line'
    selectedIndicator: 'none',  // 'none', 'bb', 'ema', 'rsi', 'macd'

    // Available timeframes
    timeframes: TIMEFRAMES,

    // Chart zoom/pan state (for persistence)
    chartVisibleRange: null,
};

const uiSlice = createSlice({
    name: 'ui',
    initialState,
    reducers: {
        setViewTimeframe: (state, action) => {
            if (TIMEFRAMES.includes(action.payload)) {
                state.viewTimeframe = action.payload;
            }
        },

        setChartType: (state, action) => {
            if (['candlestick', 'line'].includes(action.payload)) {
                state.chartType = action.payload;
            }
        },

        setSelectedIndicator: (state, action) => {
            state.selectedIndicator = action.payload;
        },

        setChartVisibleRange: (state, action) => {
            state.chartVisibleRange = action.payload;
        },
    },
});

export const {
    setViewTimeframe,
    setChartType,
    setSelectedIndicator,
    setChartVisibleRange,
} = uiSlice.actions;

// Selectors
export const selectViewTimeframe = (state) => state.ui.viewTimeframe;
export const selectChartType = (state) => state.ui.chartType;
export const selectSelectedIndicator = (state) => state.ui.selectedIndicator;
export const selectTimeframes = (state) => state.ui.timeframes;

export default uiSlice.reducer;
