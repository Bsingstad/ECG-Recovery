# Contributing to ecgrecon

Thank you for your interest in contributing to ecgrecon! This document provides guidelines and instructions for contributing.

## Ways to Contribute

- **Report bugs** by opening an issue with a clear description and minimal reproducible example
- **Suggest features** by opening an issue with a detailed explanation of the use case
- **Improve documentation** by clarifying existing docs or adding examples
- **Fix bugs or implement features** by submitting a pull request
- **Improve tests** by adding test coverage for edge cases

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/[username]/ECGRecovery.git
   cd ECGRecovery
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install in development mode with test dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Run the test suite:
   ```bash
   pytest tests/ -v
   ```

## Code Style

- Follow PEP 8
- Use meaningful variable names; prefer `ecg_signal` over `x`
- Add docstrings to functions and classes (NumPy style)
- Keep functions focused and reasonably short

## Testing

- All new code must include tests
- Tests should cover the happy path and edge cases
- Run `pytest` locally before submitting a PR
- Aim for >80% test coverage

## Submitting a Pull Request

1. Create a new branch: `git checkout -b fix/your-fix-name`
2. Make your changes and commit with clear messages
3. Push to your fork and open a pull request
4. Describe what your PR does and why
5. Link any related issues
6. Wait for review; we aim to respond within a week

## Reporting Issues

When reporting a bug, please include:
- Python version
- Operating system
- Minimal reproducible example (MRE) with code and expected vs. actual output
- Full traceback if applicable

## Code of Conduct

Please be respectful and constructive in all interactions. We expect all contributors to follow basic principles of kindness and professionalism.

## Questions?

Open an issue or discussion if you have questions about how to contribute. We're happy to help!
