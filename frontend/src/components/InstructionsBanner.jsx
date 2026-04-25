import React, { useState } from 'react';
import { Info, ChevronDown, ChevronUp, Upload, BrainCircuit, ShieldCheck } from 'lucide-react';

export default function InstructionsBanner() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="bg-white rounded-lg shadow-sm border border-blue-100 mb-6 overflow-hidden">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between p-4 bg-blue-50/50 hover:bg-blue-50 transition-colors text-left"
      >
        <div className="flex items-center text-blue-900">
          <Info className="w-5 h-5 mr-2 text-blue-600" />
          <span className="font-semibold">How to use Nexus</span>
        </div>
        {isOpen ? (
          <ChevronUp className="w-5 h-5 text-blue-500" />
        ) : (
          <ChevronDown className="w-5 h-5 text-blue-500" />
        )}
      </button>

      {isOpen && (
        <div className="p-5 border-t border-blue-100 bg-white">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            
            {/* Step 1 */}
            <div className="flex flex-col">
              <div className="flex items-center mb-2 text-gray-800">
                <div className="bg-blue-100 p-2 rounded mr-3">
                  <Upload className="w-4 h-4 text-blue-700" />
                </div>
                <h3 className="font-semibold text-sm">1. Ingest Course Materials</h3>
              </div>
              <p className="text-sm text-gray-600 ml-11">
                Upload your lecture PDFs or Slides in the <strong>Ingestion</strong> tab. Nexus will automatically chunk the text, process diagrams using Vision models, and store them in the vector database.
              </p>
            </div>

            {/* Step 2 */}
            <div className="flex flex-col">
              <div className="flex items-center mb-2 text-gray-800">
                <div className="bg-purple-100 p-2 rounded mr-3">
                  <BrainCircuit className="w-4 h-4 text-purple-700" />
                </div>
                <h3 className="font-semibold text-sm">2. Formulate Prompt</h3>
              </div>
              <p className="text-sm text-gray-600 ml-11">
                Switch to the <strong>Quiz</strong> tab and enter your request. You can specify exact topics, ask for scenario-based questions, or request a full-course review. 
              </p>
            </div>

            {/* Step 3 */}
            <div className="flex flex-col">
              <div className="flex items-center mb-2 text-gray-800">
                <div className="bg-green-100 p-2 rounded mr-3">
                  <ShieldCheck className="w-4 h-4 text-green-700" />
                </div>
                <h3 className="font-semibold text-sm">3. Review Agent Logic</h3>
              </div>
              <p className="text-sm text-gray-600 ml-11">
                As questions generate, click <strong>Agent Logic & Citation</strong> to see the exact lecture chunks the LLM used to synthesize the question and verify its accuracy.
              </p>
            </div>

          </div>
        </div>
      )}
    </div>
  );
}