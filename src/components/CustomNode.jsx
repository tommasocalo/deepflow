import React, { memo } from 'react';
import { Handle, Position } from '@xyflow/react';
import { MoreVertical, Link2 } from 'lucide-react';

const CustomNode = ({ data, id }) => {
    return (
        <div className="bg-surface border-2 border-black rounded-xl shadow-hard min-w-[280px] flex relative group transition-all hover:shadow-[6px_6px_0px_0px_#000000]">
            {/* Main Content Area */}
            <div className="flex-1 p-4">
                <h3 className="text-base font-bold tracking-tight">{data.type || 'Layer'}</h3>

                {/* Parameters - only show if not Input/Output */}
                {data.type !== 'Input' && data.type !== 'Output' && (
                    <div className="mt-3 space-y-2">
                        {data.neurons && (
                            <div className="flex justify-between items-center text-xs">
                                <span className="text-text-secondary">Neurons:</span>
                                <span className="font-bold">{data.neurons}</span>
                            </div>
                        )}
                        {data.activation && (
                            <div className="flex justify-between items-center text-xs">
                                <span className="text-text-secondary">Activation:</span>
                                <span className="font-bold">{data.activation}</span>
                            </div>
                        )}
                    </div>
                )}
            </div>

            {/* Right Action Panel */}
            <div className="flex flex-col border-l-2 border-black w-10 items-center justify-center gap-2 bg-surface-hover rounded-r-xl">
                <button className="p-1 hover:bg-black hover:text-white rounded transition-colors">
                    <MoreVertical size={16} strokeWidth={2.5} />
                </button>
                <button className="p-1 hover:bg-black hover:text-white rounded transition-colors">
                    <Link2 size={16} strokeWidth={2.5} />
                </button>
            </div>

            {/* Connection Points - Top and Bottom */}
            <Handle
                type="target"
                position={Position.Top}
                className="w-3 h-3 bg-white border-2 border-black !rounded-full hover:bg-black transition-colors"
                style={{ top: -7 }}
            />
            <Handle
                type="source"
                position={Position.Bottom}
                className="w-3 h-3 bg-white border-2 border-black !rounded-full hover:bg-black transition-colors"
                style={{ bottom: -7 }}
            />
        </div>
    );
};

export default memo(CustomNode);
