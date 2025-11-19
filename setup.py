from setuptools import setup, find_packages

setup(
    name="ml-engineer-portfolio",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        # Dependencies are in requirements.txt
    ],
    author="ML Engineer",
    description="Complete ML Engineer Portfolio Project",
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'train-model=scripts.train_model:main',
            'deploy-api=scripts.deploy_model:deploy_api',
            'monitor-drift=scripts.monitor_drift:monitor_drift',
        ],
    },
)