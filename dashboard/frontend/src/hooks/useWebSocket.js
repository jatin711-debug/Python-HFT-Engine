/**
 * WebSocket Hook with Redux Integration
 * 
 * Connects to the trading server and dispatches updates to Redux store.
 */

import { useEffect, useRef, useCallback } from 'react';
import { useDispatch } from 'react-redux';
import { setConnected, updateFromWebSocket } from '../store/tradingSlice';

export const useWebSocket = (url) => {
    const dispatch = useDispatch();
    const wsRef = useRef(null);
    const reconnectTimeoutRef = useRef(null);
    const reconnectAttempts = useRef(0);

    const connect = useCallback(() => {
        // Clear any existing connection
        if (wsRef.current) {
            wsRef.current.close();
        }

        const ws = new WebSocket(url);
        wsRef.current = ws;

        ws.onopen = () => {
            dispatch(setConnected(true));
            reconnectAttempts.current = 0;
            console.log('✅ WebSocket connected');
        };

        ws.onclose = () => {
            dispatch(setConnected(false));

            // Exponential backoff reconnection
            const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);
            reconnectAttempts.current += 1;

            console.log(`🔄 WebSocket disconnected. Reconnecting in ${delay / 1000}s...`);

            reconnectTimeoutRef.current = setTimeout(connect, delay);
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
    }, [url, dispatch]);

    useEffect(() => {
        connect();

        return () => {
            if (reconnectTimeoutRef.current) {
                clearTimeout(reconnectTimeoutRef.current);
            }
            if (wsRef.current) {
                wsRef.current.close();
            }
        };
    }, [connect]);

    const sendMessage = useCallback((msg) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(msg));
        }
    }, []);

    return { sendMessage };
};
