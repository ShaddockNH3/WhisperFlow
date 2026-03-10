import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

const apiClient = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'multipart/form-data',
    },
});

export const recognizeAudio = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    try {
        const response = await apiClient.post('/recognize', formData);
        return response.data;
    } catch (error) {
        console.error('Error recognizing audio:', error);
        throw error;
    }
};

export const deleteRecord = async (taskId) => {
    try {
        const response = await axios.delete(`${API_BASE_URL}/recognize/${taskId}`);
        return response.data;
    } catch (error) {
        console.error(`Error deleting record ${taskId}:`, error);
        throw error;
    }
};
