import React, { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud, CheckCircle, Loader2, AlertCircle, FileText } from 'lucide-react';
import { api } from '../lib/api';

export default function IngestionTab() {
  const [courseId, setCourseId] = useState('CS224');
  const [files, setFiles] = useState([]);
  const [status, setStatus] = useState('idle'); // idle, uploading, processing, complete, error
  const [errorMsg, setErrorMsg] = useState('');
  const [jobs, setJobs] = useState([]); // Array of {filename, job_id, status}

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
    setErrorMsg('');
    
    try {
      const res = await api.ingestDocuments(courseId, files);
      // res.jobs is [{filename, job_id}, ...]
      const initialJobs = res.jobs.map(j => ({ ...j, status: 'queued' }));
      setJobs(initialJobs);
      setStatus('processing');
    } catch (err) {
      setStatus('error');
      setErrorMsg(err.message);
    }
  };

  // Poll the backend for ALL job statuses
  useEffect(() => {
    let interval;
    if (status === 'processing' && jobs.length > 0) {
      interval = setInterval(async () => {
        try {
          const updatedJobs = await Promise.all(
            jobs.map(async (job) => {
              if (job.status === 'completed' || job.status.startsWith('failed')) {
                return job;
              }
              try {
                const res = await api.checkJobStatus(job.job_id);
                return { ...job, status: res.status };
              } catch (err) {
                return { ...job, status: 'failed: connection error' };
              }
            })
          );

          setJobs(updatedJobs);

          const allDone = updatedJobs.every(j => j.status === 'completed' || j.status.startsWith('failed'));
          if (allDone) {
            const anyFailed = updatedJobs.some(j => j.status.startsWith('failed'));
            setStatus(anyFailed ? 'error' : 'complete');
            clearInterval(interval);
          }
        } catch (err) {
          console.error("Polling error:", err);
        }
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [status, jobs]);

  return (
    <div className="max-w-3xl mx-auto space-y-8">
      <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
        <h2 className="text-xl font-semibold mb-4 text-gray-800">1. Define Course</h2>
        <input
          type="text"
          value={courseId}
          onChange={(e) => setCourseId(e.target.value)}
          placeholder="e.g., CS224"
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
            <div className="space-y-1">
              <p className="text-blue-600 font-medium">{files.length} file(s) selected</p>
              <div className="text-xs text-gray-400 flex flex-wrap justify-center gap-2">
                {files.map(f => <span key={f.name} className="bg-gray-100 px-2 py-1 rounded">{f.name}</span>)}
              </div>
            </div>
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
      {(status === 'processing' || jobs.length > 0) && (
        <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 space-y-4">
          <h3 className="font-semibold text-gray-800 flex items-center">
            {status === 'processing' ? <Loader2 className="w-4 h-4 mr-2 animate-spin text-blue-600" /> : <CheckCircle className="w-4 h-4 mr-2 text-green-600" />}
            Batch Ingestion Progress
          </h3>
          <div className="space-y-3">
            {jobs.map(job => (
              <div key={job.job_id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg border border-gray-100">
                <div className="flex items-center space-x-3 overflow-hidden">
                  <FileText className="w-4 h-4 text-gray-400 flex-shrink-0" />
                  <span className="text-sm text-gray-700 truncate font-medium">{job.filename}</span>
                </div>
                <div className="flex items-center space-x-2 flex-shrink-0">
                  {job.status === 'completed' ? (
                    <span className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded-full font-medium">Success</span>
                  ) : job.status.startsWith('failed') ? (
                    <span className="text-xs bg-red-100 text-red-700 px-2 py-1 rounded-full font-medium" title={job.status}>Failed</span>
                  ) : (
                    <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full font-medium flex items-center">
                      <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                      {job.status}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {status === 'complete' && (
        <div className="bg-green-50 text-green-800 p-4 rounded-lg flex items-center border border-green-100">
          <CheckCircle className="w-6 h-6 mr-3 text-green-600" />
          <p className="font-semibold">All documents processed! Vectors successfully stored in Qdrant.</p>
        </div>
      )}

      {status === 'error' && errorMsg && (
        <div className="bg-red-50 text-red-800 p-4 rounded-lg flex items-center border border-red-100">
          <AlertCircle className="w-6 h-6 mr-3 text-red-600" />
          <p className="font-semibold">Error: {errorMsg}</p>
        </div>
      )}
    </div>
  );
}