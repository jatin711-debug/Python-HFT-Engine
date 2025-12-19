/**
 * Redux Store Configuration
 * 
 * Centralized state management for the trading dashboard.
 */

import { configureStore } from '@reduxjs/toolkit';
import tradingReducer from './tradingSlice';
import uiReducer from './uiSlice';
import themeReducer from './themeSlice';

export const store = configureStore({
    reducer: {
        trading: tradingReducer,
        ui: uiReducer,
        theme: themeReducer,
    },
    middleware: (getDefaultMiddleware) =>
        getDefaultMiddleware({
            // Disable serialization check for performance with large candle arrays
            serializableCheck: false,
        }),
});

export default store;

