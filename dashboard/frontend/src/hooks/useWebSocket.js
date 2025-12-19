/**
 * WebSocket Hook with Redux Integration
 * 
 * Connects to the trading server and dispatches updates to Redux store.
 * Supports dynamic URL switching for crypto/stocks servers.
 */

import { useEffect, useRef, useCallback } from 'react';
import { useDispatch } from 'react-redux';
import { setConnected, updateFromWebSocket } from '../store/tradingSlice';

export const useWebSocket = (url) => {
    const dispatch = useDispatch();
    const wsRef = useRef(null);
    const reconnectTimeoutRef = useRef(null);
    const reconnectAttempts = useRef(0);
    const currentUrlRef = useRef(url);

    const disconnect = useCallback(() => {
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
            reconnectTimeoutRef.current = null;
        }
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        dispatch(setConnected(false));
    }, [dispatch]);

    const connect = useCallback(() => {
        // Clear any existing connection
        if (wsRef.current) {
            wsRef.current.close();
        }

        const ws = new WebSocket(currentUrlRef.current);
        wsRef.current = ws;

        ws.onopen = () => {
            dispatch(setConnected(true));
            reconnectAttempts.current = 0;
            console.log('✅ WebSocket connected to:', currentUrlRef.current);
        };

        ws.onclose = () => {
            dispatch(setConnected(false));

            // Only reconnect if we're still supposed to be connected to this URL
            if (wsRef.current === ws) {
                const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);
                reconnectAttempts.current += 1;
                console.log(`🔄 WebSocket disconnected. Reconnecting in ${delay / 1000}s...`);
                reconnectTimeoutRef.current = setTimeout(connect, delay);
            }
        };

        ws.onerror = (error) => {
            console.error('❌ WebSocket error:', error);
        };

        ws.onmessage = (event) => {
            try {
                const payload = JSON.parse(event.data);
                dispatch(updateFromWebSocket(payload));
            } catch (e) {
                console.error('Failed to parse WebSocket message:', e);
            }
        };
    }, [dispatch]);

    // Handle URL changes
    useEffect(() => {
        if (url !== currentUrlRef.current) {
            console.log('🔄 Switching WebSocket URL:', url);
            disconnect();
            currentUrlRef.current = url;
            reconnectAttempts.current = 0;
        }
        connect();

        return () => {
            disconnect();
        };
    }, [url, connect, disconnect]);

    const sendMessage = useCallback((msg) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(msg));
        }
    }, []);

    return { sendMessage, disconnect };
};

