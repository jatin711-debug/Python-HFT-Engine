/**
 * Timeframe Selector Component
 * 
 * Allows switching between different chart timeframes (view only).
 */

import { useSelector, useDispatch } from 'react-redux';
import { selectViewTimeframe, selectTimeframes, setViewTimeframe } from '../store/uiSlice';
import { Eye } from 'lucide-react';

export const TimeframeSelector = () => {
    const dispatch = useDispatch();
    const activeTimeframe = useSelector(selectViewTimeframe);
    const timeframes = useSelector(selectTimeframes);

    return (
        <div className="flex items-center gap-2">
            <div className="flex items-center gap-1 text-xs text-gray-500">
                <Eye className="w-3 h-3" />
                <span>View:</span>
            </div>
            <div className="flex gap-1 bg-bg-secondary p-1 rounded-lg border border-border">
                {timeframes.map((tf) => (
                    <button
                        key={tf}
                        onClick={() => dispatch(setViewTimeframe(tf))}
                        className={`px-2 py-1 text-xs font-medium rounded transition-all ${activeTimeframe === tf
                                ? 'bg-blue-600 text-white shadow-lg'
                                : 'text-gray-400 hover:text-white hover:bg-white/5'
                            }`}
                    >
                        {tf}
                    </button>
                ))}
            </div>
        </div>
    );
};
