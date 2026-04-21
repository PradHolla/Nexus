import React, { useState, useEffect } from 'react';
import { Loader2, Brain, ChevronDown, ChevronUp, BookOpen, FileText } from 'lucide-react';
import { api } from '../lib/api';

export default function QuizTab() {
  const [params, setParams] = useState({
    course_id: 'CS584',
    num_questions: 5,
    user_prompt: 'Generate a highly difficult, conceptual quiz focusing on LSTMs and Attention Mechanisms. Strictly ignore course administration, textbook names, and grading policies.',
    file_filters: []
  });
  
  const [availableFiles, setAvailableFiles] = useState([]);
  const [quiz, setQuiz] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [expandedQs, setExpandedQs] = useState({});

  // Fetch files whenever the course ID changes
  useEffect(() => {
    const fetchFiles = async () => {
      if (!params.course_id) return;
      try {
        const res = await api.getCourseFiles(params.course_id);
        setAvailableFiles(res.files || []);
      } catch (err) {
        console.error("Failed to fetch files", err);
      }
    };
    
    // Debounce the fetch slightly so it doesn't spam while typing
    const timeoutId = setTimeout(fetchFiles, 500);
    return () => clearTimeout(timeoutId);
  }, [params.course_id]);

  const handleFileToggle = (filename) => {
    setParams(prev => {
      const isSelected = prev.file_filters.includes(filename);
      if (isSelected) {
        return { ...prev, file_filters: prev.file_filters.filter(f => f !== filename) };
      } else {
        return { ...prev, file_filters: [...prev.file_filters, filename] };
      }
    });
  };

  const handleGenerate = async () => {
    setLoading(true);
    setError('');
    setQuiz(null);
    setExpandedQs({});

    try {
      // If no files are selected, we send null so Qdrant searches everything
      const payload = { 
        ...params, 
        file_filters: params.file_filters.length > 0 ? params.file_filters : null 
      };
      
      const result = await api.generateQuiz(payload);
      setQuiz(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const toggleLogic = (index) => {
    setExpandedQs(prev => ({ ...prev, [index]: !prev[index] }));
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
      {/* Sidebar Configuration */}
      <div className="lg:col-span-1 bg-white p-6 rounded-xl shadow-sm border border-gray-100 h-fit sticky top-8">
        <h2 className="text-xl font-semibold mb-6 flex items-center text-gray-800">
          <Brain className="w-5 h-5 mr-2 text-blue-600" />
          Agentic RAG Settings
        </h2>
        
        <div className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Course ID</label>
            <input 
              type="text" 
              value={params.course_id} 
              onChange={e => setParams({...params, course_id: e.target.value})} 
              className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 outline-none" 
            />
          </div>

          {/* Dynamic File Selector */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Source Material Boundries (Optional)</label>
            {availableFiles.length === 0 ? (
              <p className="text-xs text-gray-500 italic">No files found for this course in Qdrant.</p>
            ) : (
              <div className="max-h-40 overflow-y-auto space-y-2 border border-gray-200 p-2 rounded bg-gray-50">
                {availableFiles.map(file => (
                  <label key={file} className="flex items-center space-x-2 text-sm cursor-pointer">
                    <input 
                      type="checkbox" 
                      checked={params.file_filters.includes(file)}
                      onChange={() => handleFileToggle(file)}
                      className="rounded text-blue-600 focus:ring-blue-500"
                    />
                    <span className="text-gray-700 truncate" title={file}>{file}</span>
                  </label>
                ))}
              </div>
            )}
            <p className="text-xs text-gray-500 mt-1">Leave unchecked to allow the AI to search the entire course.</p>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Number of Questions ({params.num_questions})</label>
            <input 
              type="range" min="1" max="10" 
              value={params.num_questions} 
              onChange={e => setParams({...params, num_questions: parseInt(e.target.value)})} 
              className="w-full" 
            />
          </div>

          {/* The Agentic Prompt Area */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">AI Agent Instructions</label>
            <textarea 
              value={params.user_prompt} 
              onChange={e => setParams({...params, user_prompt: e.target.value})} 
              className="w-full p-3 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 outline-none min-h-[120px] text-sm" 
              placeholder="Tell the Planner Agent exactly what kind of quiz you want..."
            />
          </div>

          <button onClick={handleGenerate} disabled={loading} className="w-full mt-4 bg-blue-600 text-white py-3 rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 transition-all flex justify-center items-center">
            {loading ? <Loader2 className="w-5 h-5 animate-spin mr-2" /> : null}
            {loading ? 'Agents are working...' : 'Generate Quiz'}
          </button>
        </div>
      </div>

      {/* Main Quiz Area (Remains identical to your previous version) */}
      <div className="lg:col-span-2 space-y-6">
        {error && (
          <div className="bg-red-50 text-red-800 p-4 rounded-lg border border-red-100">
            {error}
          </div>
        )}

        {loading && !quiz && (
          <div className="h-64 flex flex-col items-center justify-center text-gray-500 bg-white rounded-xl border border-gray-200 border-dashed">
            <Loader2 className="w-10 h-10 animate-spin text-blue-600 mb-4" />
            <p className="font-medium text-gray-700">Executing 3-Stage Agentic Workflow...</p>
            <ul className="text-sm mt-3 space-y-1 text-gray-500 text-center">
              <li>1. Planner Agent is generating vector queries...</li>
              <li>2. Generator Agent is drafting questions using CoT...</li>
              <li>3. Critic Agent is reviewing for hallucinations...</li>
            </ul>
          </div>
        )}

        {quiz && quiz.questions.map((q, idx) => (
          <div key={idx} className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
            <div className="p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">
                <span className="text-blue-600 font-bold mr-2">Q{idx + 1}.</span> 
                {q.question_text}
              </h3>
              
              <div className="space-y-2 mb-6">
                {q.options.map((opt, i) => (
                  <div key={i} className="p-3 rounded-lg border border-gray-200 bg-gray-50 text-gray-700">
                    {String.fromCharCode(65 + i)}. {opt}
                  </div>
                ))}
              </div>

              {/* The "Show Logic" Toggle */}
              <div className="border-t border-gray-100 pt-4">
                <button 
                  onClick={() => toggleLogic(idx)}
                  className="flex items-center text-sm text-blue-600 hover:text-blue-800 font-medium transition-colors"
                >
                  {expandedQs[idx] ? <ChevronUp className="w-4 h-4 mr-1" /> : <ChevronDown className="w-4 h-4 mr-1" />}
                  {expandedQs[idx] ? 'Hide Agent Logic & Citation' : 'Show Agent Logic & Citation'}
                </button>

                {expandedQs[idx] && (
                  <div className="mt-4 p-4 bg-blue-50/50 rounded-lg border border-blue-100 space-y-4">
                    <div>
                      <span className="font-semibold text-gray-800 text-sm">Reasoning Scratchpad (Chain of Thought):</span>
                      <p className="text-gray-600 italic text-sm mt-1">"{q.reasoning_scratchpad}"</p>
                    </div>
                    <div>
                      <span className="font-semibold text-gray-800 text-sm">Correct Answer:</span>
                      <p className="text-green-700 font-medium text-sm mt-1">{q.correct_answer}</p>
                    </div>
                    <div>
                      <span className="font-semibold text-gray-800 text-sm">LLM Explanation:</span>
                      <p className="text-gray-700 text-sm mt-1">{q.explanation}</p>
                    </div>
                    <div className="flex items-center mt-3 pt-3 border-t border-blue-100">
                      <BookOpen className="w-4 h-4 text-blue-500 mr-2" />
                      <span className="text-xs font-mono text-blue-800 bg-blue-100 px-2 py-1 rounded">
                        Source Chunk ID: {q.source_chunk_id}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}