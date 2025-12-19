import { useState, useEffect, useRef } from 'react';

export const useWebSocket = (url) => {
    const [data, setData] = useState(null);
    const [connected, setConnected] = useState(false);
    const wsRef = useRef(null);

    useEffect(() => {
        const connect = () => {
            const ws = new WebSocket(url);
            wsRef.current = ws;

            ws.onopen = () => setConnected(true);
            ws.onclose = () => {
                setConnected(false);
                setTimeout(connect, 3000); // Reconnect
            };

            ws.onmessage = (event) => {
                try {
                    const payload = JSON.parse(event.data);
                    setData(payload);
                } catch (e) {
                    console.error(e);
                }
            };
        };

        connect();

        return () => {
            wsRef.current?.close();
        };
    }, [url]);

    const sendMessage = (msg) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(msg));
        }
    };

    return { data, connected, sendMessage };
};
