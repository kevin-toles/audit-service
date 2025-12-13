"""Sample code fixtures for testing auditors."""

# Example: Clean code following patterns
CLEAN_CODE_SAMPLE = '''
"""User repository implementing Repository pattern."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from sqlalchemy.orm import Session

T = TypeVar("T")


class AbstractRepository(ABC, Generic[T]):
    """Abstract repository defining the interface."""

    @abstractmethod
    def get(self, id: int) -> T | None:
        """Get entity by ID."""
        ...

    @abstractmethod
    def add(self, entity: T) -> T:
        """Add a new entity."""
        ...


class UserRepository(AbstractRepository[User]):
    """Concrete repository for User entities."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, id: int) -> User | None:
        return self.session.query(User).filter(User.id == id).first()

    def add(self, entity: User) -> User:
        self.session.add(entity)
        return entity
'''

# Example: Code with anti-patterns
CODE_WITH_ANTIPATTERNS = '''
# No docstring - missing documentation
import os, sys, json  # Multiple imports on one line

def doEverything(data, x, y, z, a, b, c, flag=True, other=None):  # Too many parameters
    # God function that does too much
    if flag:
        result = []
        for item in data:
            if item > 0:
                result.append(item * 2)
            else:
                result.append(item)
        # Magic number
        if len(result) > 100:
            result = result[:100]
    else:
        result = data
    
    # Hardcoded credentials - security issue
    password = "admin123"
    
    return result
'''

# Example: Code with security issues
CODE_WITH_SECURITY_ISSUES = '''
import os
import subprocess
from flask import request

def execute_command():
    # Command injection vulnerability
    cmd = request.args.get("cmd")
    os.system(cmd)

def sql_query(user_id):
    # SQL injection vulnerability
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return execute_sql(query)

def read_file():
    # Path traversal vulnerability
    filename = request.args.get("file")
    with open(filename, "r") as f:
        return f.read()
'''
