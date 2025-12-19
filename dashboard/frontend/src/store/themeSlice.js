/**
 * Theme Store - Dark/Light mode management
 */

import { createSlice } from '@reduxjs/toolkit';

const getInitialTheme = () => {
    if (typeof window !== 'undefined') {
        const saved = localStorage.getItem('theme');
        if (saved) return saved;
        // Check system preference
        if (window.matchMedia('(prefers-color-scheme: light)').matches) {
            return 'light';
        }
    }
    return 'dark';
};

const initialState = {
    theme: getInitialTheme(),
};

const themeSlice = createSlice({
    name: 'theme',
    initialState,
    reducers: {
        toggleTheme: (state) => {
            state.theme = state.theme === 'dark' ? 'light' : 'dark';
            if (typeof window !== 'undefined') {
                localStorage.setItem('theme', state.theme);
                document.documentElement.setAttribute('data-theme', state.theme);
            }
        },
        setTheme: (state, action) => {
            state.theme = action.payload;
            if (typeof window !== 'undefined') {
                localStorage.setItem('theme', state.theme);
                document.documentElement.setAttribute('data-theme', state.theme);
            }
        },
    },
});

export const { toggleTheme, setTheme } = themeSlice.actions;
export const selectTheme = (state) => state.theme.theme;
export default themeSlice.reducer;
