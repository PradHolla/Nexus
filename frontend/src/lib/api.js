const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

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

  // 3. Generate the actual quiz via SSE streaming
  generateQuizStream: async (params, onEvent, onError, onComplete) => {
    try {
      const response = await fetch(`${API_BASE_URL}/generate-quiz`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(params),
      });

      if (!response.ok) {
        let detail = 'Failed to generate quiz';
        try {
          const err = await response.json();
          detail = err.detail || detail;
        } catch (_) {}
        throw new Error(detail);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop() || '';

        for (const part of parts) {
          const lines = part.split('\n');
          let eventType = 'message';
          let data = {};

          for (const line of lines) {
            if (line.startsWith('event:')) {
              eventType = line.slice(6).trim();
            } else if (line.startsWith('data:')) {
              try {
                data = JSON.parse(line.slice(5).trim());
              } catch (e) {
                console.error("Error parsing SSE data:", e);
              }
            }
          }
          onEvent({ type: eventType, data });
        }
      }
      if (onComplete) onComplete();
    } catch (err) {
      if (onError) onError(err);
      else console.error("Streaming error:", err);
    }
  },

  getCourseFiles: async (courseId) => {
    const response = await fetch(`${API_BASE_URL}/courses/${courseId}/files`);
    if (!response.ok) throw new Error('Failed to fetch course files');
    return response.json();
  },

};