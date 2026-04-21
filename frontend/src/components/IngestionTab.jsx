import React, { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud, CheckCircle, Loader2, AlertCircle } from 'lucide-react';
import { api } from '../lib/api';

export default function IngestionTab() {
  const [courseId, setCourseId] = useState('CS584');
  const [files, setFiles] = useState([]);
  const [status, setStatus] = useState('idle'); // idle, uploading, processing, complete, error
  const [errorMsg, setErrorMsg] = useState('');
  const [activeJobId, setActiveJobId] = useState(null);

  const onDrop = useCallback(acceptedFiles => {
    setFiles(acceptedFiles);
    setStatus('idle');
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'], 'application/vnd.ms-powerpoint': ['.pptx'] }
  });

  const handleUpload = async () => {
    if (!courseId || files.length === 0) return;
    setStatus('uploading');
    
    try {
      const res = await api.ingestDocuments(courseId, files);
      const jobId = res.jobs[0].job_id;
      setActiveJobId(jobId);
      setStatus('processing');
    } catch (err) {
      setStatus('error');
      setErrorMsg(err.message);
    }
  };

  // Poll the backend for job status
  useEffect(() => {
    let interval;
    if (status === 'processing' && activeJobId) {
      interval = setInterval(async () => {
        try {
          const job = await api.checkJobStatus(activeJobId);
          if (job.status === 'completed') {
            setStatus('complete');
            clearInterval(interval);
          } else if (job.status.startsWith('failed')) {
            setStatus('error');
            setErrorMsg(job.status);
            clearInterval(interval);
          }
        } catch (err) {
          console.error("Polling error:", err);
        }
      }, 2000); // Check every 2 seconds
    }
    return () => clearInterval(interval);
  }, [status, activeJobId]);

  return (
    <div className="max-w-3xl mx-auto space-y-8">
      <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
        <h2 className="text-xl font-semibold mb-4 text-gray-800">1. Define Course</h2>
        <input
          type="text"
          value={courseId}
          onChange={(e) => setCourseId(e.target.value)}
          placeholder="e.g., CS584"
          className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 outline-none"
        />
      </div>

      <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
        <h2 className="text-xl font-semibold mb-4 text-gray-800">2. Upload Syllabus & Lectures</h2>
        
        <div 
          {...getRootProps()} 
          className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors ${
            isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:bg-gray-50'
          }`}
        >
          <input {...getInputProps()} />
          <UploadCloud className="mx-auto h-12 w-12 text-gray-400 mb-3" />
          {files.length > 0 ? (
            <p className="text-blue-600 font-medium">{files.length} file(s) selected</p>
          ) : (
            <p className="text-gray-500">Drag & drop PDF/PPTX files here, or click to select</p>
          )}
        </div>

        <button
          onClick={handleUpload}
          disabled={files.length === 0 || status === 'uploading' || status === 'processing'}
          className="mt-6 w-full bg-blue-600 text-white py-3 rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 transition-all flex justify-center items-center"
        >
          {status === 'uploading' && <Loader2 className="w-5 h-5 mr-2 animate-spin" />}
          {status === 'uploading' ? 'Uploading to Server...' : 'Start Ingestion Pipeline'}
        </button>
      </div>

      {/* Progress States */}
      {status === 'processing' && (
        <div className="bg-blue-50 text-blue-800 p-4 rounded-lg flex items-center border border-blue-100">
          <Loader2 className="w-6 h-6 mr-3 animate-spin text-blue-600" />
          <div>
            <p className="font-semibold">Processing Documents (Job ID: {activeJobId.slice(0,8)}...)</p>
            <p className="text-sm opacity-80">Parsing markdown, analyzing vision slides, and generating Titan embeddings.</p>
          </div>
        </div>
      )}

      {status === 'complete' && (
        <div className="bg-green-50 text-green-800 p-4 rounded-lg flex items-center border border-green-100">
          <CheckCircle className="w-6 h-6 mr-3 text-green-600" />
          <p className="font-semibold">Ingestion Complete! Vectors successfully stored in Qdrant.</p>
        </div>
      )}

      {status === 'error' && (
        <div className="bg-red-50 text-red-800 p-4 rounded-lg flex items-center border border-red-100">
          <AlertCircle className="w-6 h-6 mr-3 text-red-600" />
          <p className="font-semibold">Error: {errorMsg}</p>
        </div>
      )}
    </div>
  );
}