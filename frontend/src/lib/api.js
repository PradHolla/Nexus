const API_BASE_URL = 'http://localhost:8000/api';

export const api = {
  // 1. Upload PDFs/PPTXs to the backend
  ingestDocuments: async (courseId, files) => {
    const formData = new FormData();
    // FastAPI expects a list of files under the 'files' key
    Array.from(files).forEach(file => {
      formData.append('files', file);
    });

    const response = await fetch(`${API_BASE_URL}/ingest?course_id=${courseId}`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to start ingestion');
    }
    return response.json();
  },

  // 2. Poll the job status to show the UI loading state
  checkJobStatus: async (jobId) => {
    const response = await fetch(`${API_BASE_URL}/jobs/${jobId}`);
    if (!response.ok) throw new Error('Failed to check job status');
    return response.json();
  },

  // 3. Generate the actual quiz
  generateQuiz: async (params) => {
    const response = await fetch(`${API_BASE_URL}/generate-quiz`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params),
    });


    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to generate quiz');
    }
    return response.json();
  },

  getCourseFiles: async (courseId) => {
    const response = await fetch(`${API_BASE_URL}/courses/${courseId}/files`);
    if (!response.ok) throw new Error('Failed to fetch course files');
    return response.json();
  },

};