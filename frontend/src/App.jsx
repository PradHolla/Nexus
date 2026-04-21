import React, { useState } from 'react';
import { BookOpen, BrainCircuit } from 'lucide-react';
import IngestionTab from './components/IngestionTab';
import QuizTab from './components/QuizTab';

export default function App() {
  const [activeTab, setActiveTab] = useState('ingestion');

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 font-sans">
      {/* Top Navigation Bar */}
      <nav className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center space-x-3">
              <div className="bg-blue-600 p-2 rounded-lg">
                <BrainCircuit className="w-6 h-6 text-white" />
              </div>
              <span className="text-xl font-bold tracking-tight text-gray-900">Scholera AI</span>
            </div>
            
            <div className="flex space-x-8">
              <button
                onClick={() => setActiveTab('ingestion')}
                className={`inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors ${
                  activeTab === 'ingestion'
                    ? 'border-blue-600 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <BookOpen className="w-4 h-4 mr-2" />
                Course Materials
              </button>
              <button
                onClick={() => setActiveTab('quiz')}
                className={`inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors ${
                  activeTab === 'quiz'
                    ? 'border-blue-600 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <BrainCircuit className="w-4 h-4 mr-2" />
                Quiz Generator
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'ingestion' ? <IngestionTab /> : <QuizTab />}
      </main>
    </div>
  );
}