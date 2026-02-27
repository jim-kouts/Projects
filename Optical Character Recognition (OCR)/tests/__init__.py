# tests/__init__.py
# Empty — marks this folder as a Python package so pytest can find the tests



## Updated project structure
'''
Optical Character Recognition (OCR)/
├── .github/
│   └── workflows/
│       └── ci.yml        ← GitHub Actions pipeline
├── src/
├── api/
├── utils/
├── tests/
│   ├── __init__.py
│   └── test_utils.py     ← unit tests
├── Dockerfile
├── requirements.txt
└── README.md
'''



## What happens when you push
'''
git push
    ↓
GitHub reads .github/workflows/ci.yml
    ↓
Spins up a clean Ubuntu machine
    ↓
Runs Job 1: flake8 linting
Runs Job 2: pytest unit tests
Runs Job 3: docker build
    ↓
Shows green ✓ or red ✗ on your repo


You can see the results at:

https://github.com/jim-kouts/Projects/actions
'''