# Contributing to Three-Layer AI Framework

Thank you for your interest in contributing to the Three-Layer AI Framework! This document provides guidelines for contributing to the project.

## Code of Conduct

Be respectful, inclusive, and professional in all interactions.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/maree217/three-layer-ai-framework/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)
   - Code samples or error messages

### Suggesting Enhancements

1. Check existing issues and discussions
2. Create a new issue with:
   - Clear use case description
   - Proposed solution
   - Alternative solutions considered
   - Impact on existing functionality

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/maree217/three-layer-ai-framework
   cd three-layer-ai-framework
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the coding standards below
   - Add tests for new functionality
   - Update documentation as needed

4. **Test your changes**
   ```bash
   # Run tests
   pytest tests/

   # Check code format
   black src/ examples/
   flake8 src/ examples/

   # Type checking
   mypy src/
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` new feature
   - `fix:` bug fix
   - `docs:` documentation changes
   - `test:` test additions/changes
   - `refactor:` code refactoring
   - `perf:` performance improvements

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create Pull Request**
   - Describe your changes
   - Reference related issues
   - Include screenshots if UI changes
   - Ensure CI passes

## Development Setup

### Prerequisites
- Python 3.9+
- Git
- Azure account (for testing Azure integrations)

### Installation

```bash
# Clone repository
git clone https://github.com/maree217/three-layer-ai-framework
cd three-layer-ai-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_rag_chatbot.py

# Run with verbose output
pytest -v
```

### Code Quality

```bash
# Format code
black src/ examples/ tests/

# Lint code
flake8 src/ examples/ tests/

# Type checking
mypy src/

# Sort imports
isort src/ examples/ tests/
```

## Coding Standards

### Python Style

Follow [PEP 8](https://pep8.org/) and use type hints:

```python
# Good
def process_query(query: str, max_tokens: int = 1000) -> str:
    """
    Process a user query and return response.

    Args:
        query: User query string
        max_tokens: Maximum tokens in response

    Returns:
        Generated response string
    """
    return response

# Bad
def process_query(query, max_tokens=1000):
    return response
```

### Documentation

Use Google-style docstrings:

```python
def train_model(data: pd.DataFrame, target: str) -> Model:
    """Train a machine learning model.

    Args:
        data: Training data as pandas DataFrame
        target: Name of target column

    Returns:
        Trained model instance

    Raises:
        ValueError: If target column not found in data

    Example:
        >>> model = train_model(df, target='revenue')
        >>> predictions = model.predict(test_data)
    """
    pass
```

### Testing

Write tests for all new functionality:

```python
import pytest
from src.layer1.rag_chatbot import RAGChatbot

def test_chatbot_initialization():
    """Test chatbot initializes correctly."""
    bot = RAGChatbot(knowledge_base="./test_data")
    assert bot is not None

def test_chatbot_response():
    """Test chatbot generates response."""
    bot = RAGChatbot(knowledge_base="./test_data")
    response = bot.chat("Hello")
    assert isinstance(response, str)
    assert len(response) > 0

@pytest.mark.parametrize("query,expected_length", [
    ("short", 10),
    ("medium length query", 20),
    ("this is a much longer query that should generate a detailed response", 50)
])
def test_response_lengths(query, expected_length):
    """Test response lengths vary with query complexity."""
    bot = RAGChatbot(knowledge_base="./test_data")
    response = bot.chat(query)
    assert len(response) >= expected_length
```

## Project Structure

```
three-layer-ai-framework/
├── src/                    # Source code
│   ├── layer1/            # UX Automation
│   ├── layer2/            # Data Intelligence
│   └── layer3/            # Strategic Systems
├── examples/              # Example implementations
├── tests/                 # Test suite
├── docs/                  # Documentation
├── templates/             # Deployment templates
└── requirements.txt       # Dependencies
```

## Adding New Features

### Layer 1 (UX Automation)

1. Add implementation to `src/layer1/`
2. Add tests to `tests/layer1/`
3. Add example to `examples/`
4. Update `docs/layer1-ux-automation.md`

### Layer 2 (Data Intelligence)

1. Add implementation to `src/layer2/`
2. Add connector to `src/layer2/connectors/` if needed
3. Add tests to `tests/layer2/`
4. Update `docs/layer2-data-intelligence.md`

### Layer 3 (Strategic Intelligence)

1. Add implementation to `src/layer3/`
2. Add ML model to `src/layer3/models/` if needed
3. Add tests to `tests/layer3/`
4. Update `docs/layer3-strategic-systems.md`

## Documentation

### README Updates

Keep README.md concise. Detailed docs go in `docs/`.

### API Documentation

Update `docs/api.md` for new public APIs.

### Examples

Add working examples to `examples/` directory with:
- README.md explaining the example
- Sample data (or instructions to generate it)
- Expected output

## Release Process

Maintainers will:

1. Update version in `setup.py`
2. Update CHANGELOG.md
3. Create git tag
4. Build and publish to PyPI
5. Create GitHub release

## Getting Help

- 📖 Read the [documentation](docs/)
- 💬 Open a [GitHub Discussion](https://github.com/maree217/three-layer-ai-framework/discussions)
- 🐛 Report bugs via [Issues](https://github.com/maree217/three-layer-ai-framework/issues)
- 📧 Email: 2maree@gmail.com

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing!** 🎉
