#!/usr/bin/env python3
"""
Model Deployment Script
"""
import sys
import os
import uvicorn

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def deploy_api():
    """Deploy the FastAPI application"""
    logger.info("Starting model deployment")
    
    api_config = config.get_api_config().get('server', {})
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 8000)
    reload = api_config.get('reload', True)
    
    logger.info(f"Starting API server on {host}:{port}")
    logger.info(f"Reload mode: {reload}")
    
    uvicorn.run(
        "src.deployment.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    deploy_api()