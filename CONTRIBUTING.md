# Contributing to Qwen3 TTS

We welcome contributions! Here's how to get started.

## Code of Conduct

By participating, you agree to maintain a respectful and inclusive environment.

## How to Contribute

### Reporting Bugs

1. Check existing issues to avoid duplicates
2. Include your platform (macOS/Windows), Python version, and GPU info
3. Provide steps to reproduce, expected vs actual behavior
4. Attach terminal output/logs if applicable

### Feature Requests

1. Describe the feature and its use case
2. Explain how it integrates with the existing architecture
3. Suggest implementation approach if possible

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/amazing-feature`)
3. Make your changes
4. Run the test suite: `python test_suite.py`
5. Commit with clear messages
6. Open a PR against `main`

### Development Setup

```bash
git clone https://github.com/YOUR_USER/qwen3-tts-ui
cd qwen3-tts-ui
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
```

### Code Style

- Python: PEP 8, max line length 100
- JavaScript: Standard style
- HTML: semantic, accessible
- Comments: meaningful, not redundant

### Testing

- All new features should include tests
- Run `python test_suite.py` before submitting
- Ensure no regressions on existing functionality

### Documentation

- Update README.md for user-facing changes
- Update docs/ for architectural changes
- Add inline comments for non-obvious logic

## Review Process

1. Maintainers review within 1 week
2. CI must pass
3. At least one approval required
4. Squash merge into main

## License

By contributing, you agree that your contributions will be licensed under Apache 2.0.
