import pytest
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.deployment.api import app
from fastapi.testclient import TestClient

class TestDeployment:
    def test_api_health(self):
        """Test API health endpoint"""
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data
    
    def test_api_root(self):
        """Test API root endpoint"""
        client = TestClient(app)
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "status" in data
    
    def test_models_endpoint(self):
        """Test models list endpoint"""
        client = TestClient(app)
        response = client.get("/models")
        
        assert response.status_code == 200
        data = response.json()
        assert "available_models" in data

if __name__ == "__main__":
    pytest.main([__file__, "-v"])