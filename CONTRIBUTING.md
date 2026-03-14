# Contributing to TemporalCorr-MetaNet

Thank you for your interest in contributing to TemporalCorr-MetaNet!

## Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/username/TemporalCorr-MetaNet.git
cd TemporalCorr-MetaNet
pip install -e ".[dev]"
```

## Code Style

We use:
- **Black** for formatting (line length: 100)
- **isort** for import sorting
- **flake8** for linting

```bash
# Format code
black .
isort .

# Check linting
flake8 .
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit with clear messages
6. Push and create a Pull Request

## Reporting Issues

Please include:
- Python version
- PyTorch version
- Steps to reproduce
- Expected vs actual behavior
- Error messages/stack traces
