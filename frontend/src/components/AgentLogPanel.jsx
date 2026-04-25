import React, { useEffect, useRef, useState } from 'react';
import { Terminal, ChevronUp, ChevronDown, Activity } from 'lucide-react';

export default function AgentLogPanel({ logs, isGenerating, title = "Backend Logs" }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs, isExpanded]);

  useEffect(() => {
    if (isGenerating) setIsExpanded(true);
  }, [isGenerating]);

  return (
    <div className="mt-6 border border-gray-800 rounded-lg overflow-hidden bg-gray-900 shadow-lg font-sans">
      <div 
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center justify-between px-4 py-3 bg-gray-800 cursor-pointer hover:bg-gray-700 transition-colors"
      >
        <div className="flex items-center space-x-2 text-gray-300">
          <Terminal className="w-5 h-5" />
          <span className="font-semibold text-sm tracking-wide">{title}</span>
          {isGenerating && <Activity className="w-4 h-4 ml-2 text-green-400 animate-pulse" />}
        </div>
        <div className="flex items-center space-x-2 text-gray-400">
          <span className="text-xs">{logs.length} events</span>
          {isExpanded ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
        </div>
      </div>

      {isExpanded && (
        <div 
          ref={scrollRef}
          className="p-4 max-h-80 overflow-y-auto font-mono text-xs bg-gray-900 space-y-3 scrollbar-thin scrollbar-thumb-gray-600"
        >
          {logs.length === 0 ? (
            <div className="text-gray-500 italic">Waiting for backend events...</div>
          ) : (
            logs.map((log, index) => (
              <div key={index} className="flex flex-col space-y-1 border-b border-gray-800 pb-2">
                <div className="flex space-x-3 items-center">
                  <span className="text-gray-500">[{log.timestamp}]</span>
                  <span className="font-bold text-blue-400">event: {log.event}</span>
                </div>
                <div className="pl-16 text-gray-300 whitespace-pre-wrap">
                  data: {JSON.stringify(log.data, null, 2)}
                </div>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}