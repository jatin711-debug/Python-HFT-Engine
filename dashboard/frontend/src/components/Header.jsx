import { Activity } from 'lucide-react';

export const Header = ({ connected }) => (
    <div className="flex items-center gap-3">
        <div className="bg-blue-600/20 p-2 rounded-lg border border-blue-500/30">
            <Activity className="w-6 h-6 text-blue-400" />
        </div>
        <div>
            <h1 className="text-xl font-bold tracking-tight">ProHFT Terminal</h1>
            <div className="flex items-center gap-2 text-xs text-gray-400">
                <span className={`w-2 h-2 rounded-full ${connected ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]' : 'bg-red-500'}`} />
                {connected ? 'System Online' : 'Reconnecting...'}
            </div>
        </div>
    </div>
);
