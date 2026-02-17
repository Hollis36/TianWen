# Contributing to TianWen

Thank you for your interest in contributing to TianWen! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the project and community

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/TianWen.git
cd TianWen

# Add upstream remote
git remote add upstream https://github.com/Hollis36/TianWen.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### 3. Create a Branch

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Or a bugfix branch
git checkout -b fix/issue-description
```

## Development Guidelines

### Code Style

We use automated code formatting and linting:

- **Black** for code formatting (line length: 100)
- **isort** for import sorting
- **flake8** for linting
- **pydocstyle** for docstring conventions (Google style)

Run formatting before committing:

```bash
# Format code
black tianwen/ tests/ tools/

# Sort imports
isort tianwen/ tests/ tools/

# Check linting
flake8 tianwen/ tests/ tools/
```

Or use pre-commit hooks (recommended):

```bash
# Run on all files
pre-commit run --all-files

# Runs automatically on git commit after installation
```

### Type Hints

- Add type hints to all function signatures
- Use `typing` module for complex types
- Example:
  ```python
  from typing import List, Dict, Optional
  
  def process_data(
      data: List[Dict[str, Any]],
      threshold: float = 0.5,
      config: Optional[Dict] = None
  ) -> List[Dict[str, Any]]:
      ...
  ```

### Docstrings

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """
    Brief description of the function.

    More detailed explanation if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: If param2 is negative

    Example:
        >>> example_function("test", 10)
        True
    """
    ...
```

### Testing

- Write tests for all new functionality
- Maintain or improve code coverage
- Use pytest for testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_validation.py -v

# Run with coverage
pytest tests/ --cov=tianwen --cov-report=html
```

### Error Handling

- Use custom exception classes from `tianwen.utils.errors`
- Provide informative error messages
- Include context and suggestions when appropriate

```python
from tianwen.utils.errors import ValidationError

if value < 0:
    raise ValidationError(
        "Value must be non-negative",
        field_name="batch_size",
        invalid_value=value
    )
```

### Validation

- Use validation utilities from `tianwen.utils.validation`
- Validate inputs early in functions
- Provide clear error messages

```python
from tianwen.utils.validation import validate_image_tensor

def process_image(image: Tensor) -> Tensor:
    validate_image_tensor(image)  # Validates shape, dtype, etc.
    # ... process image
```

## Adding New Components

### Adding a Detector

1. Create detector class inheriting from `BaseDetector`
2. Implement all abstract methods
3. Register with decorator: `@DETECTORS.register("my_detector")`
4. Add configuration in `configs/detector/`
5. Add tests in `tests/`

Example:
```python
from tianwen.core.registry import DETECTORS
from tianwen.detectors.base import BaseDetector

@DETECTORS.register("my_detector")
class MyDetector(BaseDetector):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(num_classes, **kwargs)
        # Initialize your detector
    
    def forward(self, images, targets=None):
        # Implement forward pass
        pass
    
    def extract_features(self, images, feature_levels=None):
        # Implement feature extraction
        pass
    
    def compute_loss(self, predictions, targets):
        # Implement loss computation
        pass
    
    def get_optimizer_groups(self, lr, weight_decay=0.0):
        # Return parameter groups
        pass
```

### Adding a VLM

Similar process for VLMs - inherit from `BaseVLM` and register with `@VLMS.register()`.

### Adding a Fusion Strategy

Inherit from `BaseFusion` and register with `@FUSIONS.register()`.

## Pull Request Process

### Before Submitting

1. Update your branch with upstream:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. Run all checks:
   ```bash
   # Format code
   black tianwen/ tests/ tools/
   isort tianwen/ tests/ tools/
   
   # Run linting
   flake8 tianwen/ tests/ tools/
   
   # Run tests
   pytest tests/ -v
   
   # Or use pre-commit
   pre-commit run --all-files
   ```

3. Update documentation if needed

### Submitting PR

1. Push your branch to your fork:
   ```bash
   git push origin your-branch-name
   ```

2. Create PR on GitHub with:
   - Clear title describing the change
   - Description explaining what and why
   - Reference any related issues
   - Screenshots for UI changes

3. Wait for review and address feedback

### PR Review Checklist

Your PR should:
- [ ] Pass all CI checks
- [ ] Include tests for new functionality
- [ ] Update documentation if needed
- [ ] Follow code style guidelines
- [ ] Have clear commit messages
- [ ] Not introduce breaking changes (without discussion)

## Commit Message Guidelines

Use conventional commits format:

```
type(scope): brief description

Longer description if needed.

Fixes #123
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(detectors): add YOLOv11 detector support

fix(fusion): correct feature dimension mismatch in distillation

docs(readme): add troubleshooting section

test(validation): add tests for box validation
```

## Release Process

(For maintainers)

1. Update version in `tianwen/__init__.py` and `pyproject.toml`
2. Update CHANGELOG.md
3. Create release tag: `git tag -a v0.2.0 -m "Release v0.2.0"`
4. Push tag: `git push origin v0.2.0`
5. Create GitHub release with notes

## Getting Help

- **Issues**: Use GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub discussions for questions
- **Documentation**: Check docs/ folder

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Credited in release notes
- Mentioned in commit co-author tags

Thank you for contributing to TianWen! 🚀
