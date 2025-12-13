# audit-service

**Kitchen Brigade Role**: 🔍 Auditor  
**Port**: 8084  
**Status**: Scaffolding  

---

## Overview

The **audit-service** is a dedicated microservice for validating code against reference materials, detecting anti-patterns, and ensuring compliance with coding standards. It serves as the "Auditor" in the Kitchen Brigade architecture.

## Responsibilities

- **Pattern Compliance**: Validate code against architectural patterns from reference materials
- **Anti-Pattern Detection**: Scan for known code smells from CODING_PATTERNS_ANALYSIS.md (252 patterns) and Comp_Static_Analysis_Report (52 issues)
- **Security Scanning**: OWASP compliance, secret exposure detection
- **Citation Verification**: Verify that generated code citations match source materials
- **Audit Checkpoints**: Three-stage validation for agentic code editing pipeline

## Kitchen Brigade Integration

```
┌─────────────────────────────────────────────────────────────────┐
│  🔍 AUDITOR (audit-service, Port 8084)                         │
│                                                                 │
│  Called by: ai-agents (Expeditor)                              │
│  Inputs: Code + Reference Materials                            │
│  Outputs: AuditResult (passed, score, violations, suggestions) │
│                                                                 │
│  Checkpoints:                                                   │
│  • #1 Post-Draft: Validates CodeT5+ output                     │
│  • #2 Post-LLM: Validates LLM implementation                   │
│  • #3 Post-External: Final validation before delivery          │
└─────────────────────────────────────────────────────────────────┘
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness probe |
| GET | `/health/ready` | Readiness probe |
| POST | `/v1/audit` | Full audit of code against references |
| POST | `/v1/audit/checkpoint/1` | Post-draft validation |
| POST | `/v1/audit/checkpoint/2` | Post-LLM validation |
| POST | `/v1/audit/checkpoint/3` | Final validation |

## Quick Start

```bash
# Install dependencies
poetry install

# Run development server
poetry run uvicorn src.main:app --reload --port 8084

# Run tests
poetry run pytest tests/ -v

# Build Docker image
docker build -f deploy/docker/Dockerfile -t audit-service:latest .
```

## Directory Structure

```
audit-service/
├── src/
│   ├── api/routes/          # API endpoints
│   ├── core/                # Config, exceptions
│   ├── auditors/            # Audit implementations
│   ├── rules/               # Rule definitions
│   ├── models/              # Request/response models
│   └── main.py              # FastAPI app
├── tests/
│   ├── unit/                # Unit tests
│   ├── integration/         # API tests
│   └── fixtures/            # Test data
├── config/                  # Rule configurations
├── deploy/                  # Docker, Kubernetes
└── .github/workflows/       # CI/CD
```

## Anti-Pattern Coverage

This service enforces patterns from:
- `CODING_PATTERNS_ANALYSIS.md`: 252 categorized issues
- `Comp_Static_Analysis_Report_20251203.md`: 52 resolved issues

Key categories:
- Race conditions (#9-11)
- Exception shadowing (#6-7)
- Connection pooling (#12)
- Secret exposure (#17-19)
- Cognitive complexity (Category 2)

## Development

```bash
# Format code
poetry run black src/ tests/
poetry run ruff check --fix .

# Type checking
poetry run mypy src/

# Run specific test
poetry run pytest tests/unit/test_auditors/ -v
```

## License

MIT
