import React from 'react';
import ReactDOM from 'react-dom/client';
import { Provider } from 'react-redux';
import store from './store';
import App from './App.jsx';
import './index.css';

// Initialize theme from localStorage before render
const savedTheme = localStorage.getItem('theme') || 'dark';
document.documentElement.setAttribute('data-theme', savedTheme);

// Note: StrictMode removed to prevent WebSocket double-connections in dev
ReactDOM.createRoot(document.getElementById('root')).render(
    <Provider store={store}>
        <App />
    </Provider>
);

