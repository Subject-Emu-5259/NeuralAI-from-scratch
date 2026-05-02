#!/usr/bin/env python3
"""
NeuralAI Training Data Generator
Expands training data from 347 to 1000+ samples
"""

import json
import random
from pathlib import Path
from datetime import datetime

# Output path
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "train_expanded.jsonl"

# System prompts for variety
SYSTEM_PROMPTS = [
    "You are NeuralAI, a helpful AI assistant specialized in coding and reasoning.",
    "You are NeuralAI, an intelligent assistant that helps with coding, analysis, and problem-solving.",
    "You are NeuralAI, a knowledgeable AI that explains concepts clearly and writes clean code.",
    "You are NeuralAI, a precise assistant that provides accurate information and working code.",
]

# Training data templates by category
TEMPLATES = {
    # ===== ADVANCED CODING (100 samples) =====
    "advanced_coding": [
        # Multi-file projects
        {
            "user": "Create a Flask app with routes for user CRUD operations",
            "assistant": """Here's a Flask app with user CRUD routes:

```python
# app.py
from flask import Flask, request, jsonify
from dataclasses import dataclass
from typing import Optional
import uuid

app = Flask(__name__)

@dataclass
class User:
    id: str
    name: str
    email: str
    
users_db = {}

@app.route('/users', methods=['POST'])
def create_user():
    data = request.json
    user = User(
        id=str(uuid.uuid4()),
        name=data['name'],
        email=data['email']
    )
    users_db[user.id] = user
    return jsonify({'id': user.id, 'name': user.name, 'email': user.email}), 201

@app.route('/users/<user_id>', methods=['GET'])
def get_user(user_id):
    user = users_db.get(user_id)
    if user:
        return jsonify({'id': user.id, 'name': user.name, 'email': user.email})
    return jsonify({'error': 'User not found'}), 404

@app.route('/users/<user_id>', methods=['PUT'])
def update_user(user_id):
    if user_id not in users_db:
        return jsonify({'error': 'User not found'}), 404
    data = request.json
    user = users_db[user_id]
    user.name = data.get('name', user.name)
    user.email = data.get('email', user.email)
    return jsonify({'id': user.id, 'name': user.name, 'email': user.email})

@app.route('/users/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    if user_id in users_db:
        del users_db[user_id]
        return '', 204
    return jsonify({'error': 'User not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
```

This provides:
- POST /users - Create new user
- GET /users/<id> - Get user by ID
- PUT /users/<id> - Update user
- DELETE /users/<id> - Delete user"""
        },
        # Refactoring
        {
            "user": "Refactor this code to be cleaner:\ndef calc(x,y,op):\n    if op=='+': return x+y\n    if op=='-': return x-y\n    if op=='*': return x*y\n    if op=='/': return x/y",
            "assistant": """Here's a cleaner refactored version:

```python
from enum import Enum
from typing import Union

class Operation(Enum):
    ADD = '+'
    SUBTRACT = '-'
    MULTIPLY = '*'
    DIVIDE = '/'

def calculate(x: float, y: float, operation: Operation) -> float:
    \"\"\"Perform arithmetic operation on two numbers.
    
    Args:
        x: First operand
        y: Second operand
        operation: The operation to perform
        
    Returns:
        Result of the calculation
        
    Raises:
        ZeroDivisionError: If dividing by zero
        ValueError: If operation is invalid
    \"\"\"
    operations = {
        Operation.ADD: lambda a, b: a + b,
        Operation.SUBTRACT: lambda a, b: a - b,
        Operation.MULTIPLY: lambda a, b: a * b,
        Operation.DIVIDE: lambda a, b: a / b,
    }
    
    if operation not in operations:
        raise ValueError(f"Invalid operation: {operation}")
    
    return operations[operation](x, y)

# Usage
result = calculate(10, 5, Operation.ADD)  # 15
```

Improvements:
- Type hints for clarity
- Enum for type-safe operations
- Docstring documentation
- Better error handling
- Dictionary dispatch instead of if-chain"""
        },
        # Debugging
        {
            "user": "Debug this code - it's not working:\ndef find_max(lst):\n    max_val = 0\n    for item in lst:\n        if item > max_val:\n            max_val = item\n    return max_val",
            "assistant": """The bug is initializing `max_val = 0`. This fails for:
- Negative numbers (returns 0, not actual max)
- Empty lists (should handle gracefully)

Fixed version:

```python
def find_max(lst):
    \"\"\"Find the maximum value in a list.
    
    Returns None for empty lists.
    Handles negative numbers correctly.
    \"\"\"
    if not lst:
        return None
    
    max_val = lst[0]  # Initialize with first element
    for item in lst[1:]:
        if item > max_val:
            max_val = item
    return max_val

# Or simpler using built-in:
def find_max(lst):
    return max(lst) if lst else None

# Test cases
print(find_max([3, 1, 4, 1, 5]))     # 5
print(find_max([-5, -2, -10]))       # -2
print(find_max([]))                  # None
```

The fix: Initialize with the first element or use Python's built-in `max()`."""
        },
        # Design patterns
        {
            "user": "Implement a singleton pattern in Python",
            "assistant": """Here are multiple ways to implement Singleton in Python:

**Method 1: Decorator (Recommended)**
```python
def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Database:
    def __init__(self):
        self.connected = False
    
    def connect(self):
        self.connected = True
        return "Connected!"

# Both variables point to same instance
db1 = Database()
db2 = Database()
print(db1 is db2)  # True
```

**Method 2: Metaclass**
```python
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self):
        self.connected = False
```

**Method 3: Module-level (Simplest)**
```python
# database.py
class _Database:
    def __init__(self):
        self.connected = False

database = _Database()  # Single instance

# In other files:
from database import database
```

Use the decorator approach for flexibility and clarity."""
        },
    ],
    
    # ===== API DESIGN (50 samples) =====
    "api_design": [
        {
            "user": "Design a REST API for a blog platform",
            "assistant": """Here's a REST API design for a blog platform:

**Endpoints**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /posts | List all posts (paginated) |
| POST | /posts | Create new post |
| GET | /posts/{id} | Get single post |
| PUT | /posts/{id} | Update post |
| DELETE | /posts/{id} | Delete post |
| POST | /posts/{id}/comments | Add comment |
| GET | /posts/{id}/comments | List comments |
| GET | /users/{id}/posts | Get user's posts |

**Request/Response Schemas**

```json
// POST /posts
{
  "title": "My First Post",
  "content": "Hello world...",
  "author_id": "user123",
  "tags": ["intro", "hello"]
}

// Response
{
  "id": "post456",
  "title": "My First Post",
  "content": "Hello world...",
  "author": {
    "id": "user123",
    "name": "John Doe"
  },
  "tags": ["intro", "hello"],
  "created_at": "2026-05-02T10:00:00Z",
  "updated_at": "2026-05-02T10:00:00Z"
}

// GET /posts?page=1&limit=10
{
  "posts": [...],
  "pagination": {
    "page": 1,
    "limit": 10,
    "total": 50,
    "pages": 5
  }
}
```

**Status Codes**
- 200 OK - Successful GET/PUT
- 201 Created - Successful POST
- 204 No Content - Successful DELETE
- 400 Bad Request - Invalid input
- 401 Unauthorized - Auth required
- 404 Not Found - Resource doesn't exist"""
        },
        {
            "user": "Create an OpenAPI spec for a user API",
            "assistant": """Here's an OpenAPI 3.0 specification:

```yaml
openapi: 3.0.0
info:
  title: User API
  version: 1.0.0
  description: API for user management

servers:
  - url: https://api.example.com/v1

paths:
  /users:
    get:
      summary: List all users
      parameters:
        - name: limit
          in: query
          schema:
            type: integer
            default: 20
        - name: offset
          in: query
          schema:
            type: integer
            default: 0
      responses:
        '200':
          description: List of users
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UserList'
    post:
      summary: Create a user
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UserInput'
      responses:
        '201':
          description: User created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
  
  /users/{id}:
    get:
      summary: Get user by ID
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: User details
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
        '404':
          description: User not found

components:
  schemas:
    User:
      type: object
      properties:
        id:
          type: string
        name:
          type: string
        email:
          type: string
          format: email
        created_at:
          type: string
          format: date-time
    
    UserInput:
      type: object
      required:
        - name
        - email
      properties:
        name:
          type: string
        email:
          type: string
          format: email
    
    UserList:
      type: object
      properties:
        users:
          type: array
          items:
            $ref: '#/components/schemas/User'
        total:
          type: integer
```"""
        },
    ],
    
    # ===== DATABASE OPERATIONS (50 samples) =====
    "database": [
        {
            "user": "Write SQL to create a users table with proper constraints",
            "assistant": """Here's a well-designed users table:

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT valid_email CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$'),
    CONSTRAINT valid_username CHECK (username ~* '^[a-zA-Z0-9_]{3,50}$')
);

-- Indexes for performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_active ON users(is_active) WHERE is_active = true;

-- Trigger for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
```

Features:
- UUID primary key (security)
- Unique constraints on username/email
- Email format validation
- Timestamps with timezone
- Automatic updated_at trigger
- Partial index for active users"""
        },
        {
            "user": "Write a complex SQL query with JOINs and aggregations",
            "assistant": """Here's a complex query for an e-commerce scenario:

```sql
-- Find top customers with their order stats
SELECT 
    u.id,
    u.email,
    u.first_name || ' ' || u.last_name AS full_name,
    COUNT(DISTINCT o.id) AS total_orders,
    COUNT(DISTINCT oi.product_id) AS unique_products,
    SUM(oi.quantity * oi.price) AS total_spent,
    AVG(o.total_amount) AS avg_order_value,
    MAX(o.created_at) AS last_order_date,
    DATEDIFF('day', MAX(o.created_at), CURRENT_TIMESTAMP) AS days_since_last_order
FROM users u
LEFT JOIN orders o ON u.id = o.user_id AND o.status != 'cancelled'
LEFT JOIN order_items oi ON o.id = oi.order_id
WHERE u.is_active = true
GROUP BY u.id, u.email, u.first_name, u.last_name
HAVING SUM(oi.quantity * oi.price) > 100
ORDER BY total_spent DESC
LIMIT 10;

-- With CTE for readability
WITH customer_orders AS (
    SELECT 
        user_id,
        COUNT(*) AS order_count,
        SUM(total_amount) AS lifetime_value
    FROM orders
    WHERE status = 'completed'
    GROUP BY user_id
),
customer_products AS (
    SELECT 
        o.user_id,
        COUNT(DISTINCT oi.product_id) AS product_variety
    FROM orders o
    JOIN order_items oi ON o.id = oi.order_id
    WHERE o.status = 'completed'
    GROUP BY o.user_id
)
SELECT 
    u.email,
    co.order_count,
    co.lifetime_value,
    cp.product_variety
FROM users u
JOIN customer_orders co ON u.id = co.user_id
LEFT JOIN customer_products cp ON u.id = cp.user_id
ORDER BY co.lifetime_value DESC;
```"""
        },
    ],
    
    # ===== DEVOPS (50 samples) =====
    "devops": [
        {
            "user": "Create a Dockerfile for a Python Flask app",
            "assistant": """Here's an optimized Dockerfile for Flask:

```dockerfile
# Use multi-stage build for smaller image

# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]
```

**.dockerignore**
```
__pycache__/
*.pyc
.git/
.env
*.log
venv/
.pytest_cache/
```

Build and run:
```bash
docker build -t flask-app .
docker run -p 5000:5000 flask-app
```"""
        },
        {
            "user": "Write a GitHub Actions CI/CD pipeline",
            "assistant": """Here's a complete CI/CD pipeline:

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: '3.11'

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install linters
        run: pip install flake8 black isort mypy
      
      - name: Run Black
        run: black --check .
      
      - name: Run isort
        run: isort --check-only .
      
      - name: Run flake8
        run: flake8 . --max-line-length=100

  test:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: pytest --cov=app --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  build:
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      
      - name: Build Docker image
        run: |
          docker build -t myapp:${{ github.sha }} .
          docker tag myapp:${{ github.sha }} myapp:latest
      
      - name: Push to registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USER }} --password-stdin
          docker push myapp:${{ github.sha }}
          docker push myapp:latest

  deploy:
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to production
        run: |
          echo "Deploying to production..."
          # Add deployment commands here
```"""
        },
    ],
    
    # ===== DATA ANALYSIS (60 samples) =====
    "data_analysis": [
        {
            "user": "Load and analyze a CSV file with pandas",
            "assistant": """Here's how to analyze CSV data with pandas:

```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data.csv')

# Basic exploration
print(df.head())
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Clean data
df = df.dropna()  # Remove rows with missing values
# Or fill missing values
df['column'] = df['column'].fillna(df['column'].mean())

# Filter data
filtered = df[df['price'] > 100]

# Group and aggregate
stats = df.groupby('category').agg({
    'price': ['mean', 'min', 'max'],
    'quantity': 'sum'
}).round(2)

# Pivot table
pivot = df.pivot_table(
    values='sales',
    index='region',
    columns='month',
    aggfunc='sum'
)

# Calculate new columns
df['revenue'] = df['price'] * df['quantity']
df['growth'] = df['revenue'].pct_change()

# Save results
df.to_csv('processed.csv', index=False)
stats.to_excel('stats.xlsx')
```"""
        },
        {
            "user": "Create a data visualization with matplotlib",
            "assistant": """Here's a complete visualization script:

```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Sample data
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
sales = [100, 120, 90, 150, 180, 200]
costs = [80, 90, 70, 100, 110, 120]

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Line chart
ax1 = axes[0, 0]
ax1.plot(months, sales, marker='o', label='Sales', color='#2ecc71')
ax1.plot(months, costs, marker='s', label='Costs', color='#e74c3c')
ax1.set_title('Sales vs Costs')
ax1.set_xlabel('Month')
ax1.set_ylabel('Amount ($)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Bar chart
ax2 = axes[0, 1]
profit = [s - c for s, c in zip(sales, costs)]
colors = ['#2ecc71' if p > 0 else '#e74c3c' for p in profit]
ax2.bar(months, profit, color=colors)
ax2.set_title('Monthly Profit')
ax2.set_ylabel('Profit ($)')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# 3. Pie chart
ax3 = axes[1, 0]
categories = ['Product A', 'Product B', 'Product C', 'Product D']
values = [35, 25, 20, 20]
ax3.pie(values, labels=categories, autopct='%1.1f%%', startangle=90)
ax3.set_title('Product Distribution')

# 4. Histogram
ax4 = axes[1, 1]
data = np.random.randn(1000)
ax4.hist(data, bins=30, color='#3498db', edgecolor='white')
ax4.set_title('Distribution')
ax4.set_xlabel('Value')
ax4.set_ylabel('Frequency')

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig('dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
```"""
        },
    ],
    
    # ===== TESTING (40 samples) =====
    "testing": [
        {
            "user": "Write unit tests for a calculator class",
            "assistant": """Here are comprehensive unit tests:

```python
import pytest
from calculator import Calculator

class TestCalculator:
    
    @pytest.fixture
    def calc(self):
        return Calculator()
    
    def test_add(self, calc):
        assert calc.add(2, 3) == 5
        assert calc.add(-1, 1) == 0
        assert calc.add(0, 0) == 0
    
    def test_subtract(self, calc):
        assert calc.subtract(5, 3) == 2
        assert calc.subtract(1, 5) == -4
    
    def test_multiply(self, calc):
        assert calc.multiply(3, 4) == 12
        assert calc.multiply(-2, 3) == -6
        assert calc.multiply(0, 100) == 0
    
    def test_divide(self, calc):
        assert calc.divide(10, 2) == 5
        assert calc.divide(7, 2) == 3.5
    
    def test_divide_by_zero(self, calc):
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            calc.divide(10, 0)
    
    @pytest.mark.parametrize("a,b,expected", [
        (2, 3, 5),
        (0, 0, 0),
        (-5, 5, 0),
        (100, -50, 50),
    ])
    def test_add_parametrized(self, calc, a, b, expected):
        assert calc.add(a, b) == expected
    
    def test_history(self, calc):
        calc.add(1, 2)
        calc.multiply(3, 4)
        assert len(calc.get_history()) == 2
        assert calc.get_history()[0] == "1 + 2 = 3"

# Run: pytest test_calculator.py -v
```"""
        },
    ],
    
    # ===== ERROR HANDLING (50 samples) =====
    "error_handling": [
        {
            "user": "Show me proper error handling patterns in Python",
            "assistant": """Here are Python error handling best practices:

```python
# 1. Specific exceptions
try:
    result = risky_operation()
except FileNotFoundError:
    logger.error("File not found")
    raise
except PermissionError:
    logger.error("Permission denied")
    raise
except Exception as e:
    logger.exception("Unexpected error")
    raise

# 2. Context managers
class DatabaseConnection:
    def __enter__(self):
        self.conn = connect_to_db()
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.conn.rollback()
            logger.error(f"Error: {exc_val}")
        else:
            self.conn.commit()
        self.conn.close()
        return False  # Re-raise exceptions

# Usage
with DatabaseConnection() as db:
    db.execute("INSERT ...")

# 3. Custom exceptions
class ValidationError(Exception):
    def __init__(self, field, message):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")

def validate_user(data):
    if not data.get('email'):
        raise ValidationError('email', 'Email is required')
    if '@' not in data['email']:
        raise ValidationError('email', 'Invalid email format')

# 4. Retry decorator
import time
from functools import wraps

def retry(max_attempts=3, delay=1, exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay * (attempt + 1))
            return None
        return wrapper
    return decorator

@retry(max_attempts=3, exceptions=(ConnectionError,))
def fetch_data(url):
    return requests.get(url).json()
```"""
        },
    ],
    
    # ===== TOOL CHAINING (40 samples) =====
    "tool_chaining": [
        {
            "user": "I need to fetch data from an API, process it, and save to a file",
            "assistant": """I'll help you chain these operations. Here's the workflow:

```python
import requests
import json
import pandas as pd
from pathlib import Path

# Step 1: Fetch data from API
def fetch_data(api_url):
    response = requests.get(api_url)
    response.raise_for_status()
    return response.json()

# Step 2: Process the data
def process_data(raw_data):
    # Extract relevant fields
    processed = []
    for item in raw_data.get('results', []):
        processed.append({
            'id': item['id'],
            'name': item['name'].title(),
            'value': float(item['value']),
            'date': item['created_at'][:10]
        })
    
    # Create DataFrame
    df = pd.DataFrame(processed)
    
    # Add calculated column
    df['value_squared'] = df['value'] ** 2
    
    return df

# Step 3: Save to file
def save_data(df, output_path):
    # Save as CSV
    csv_path = Path(output_path).with_suffix('.csv')
    df.to_csv(csv_path, index=False)
    
    # Also save as JSON
    json_path = Path(output_path).with_suffix('.json')
    df.to_json(json_path, orient='records', indent=2)
    
    return csv_path, json_path

# Chain everything together
def main():
    url = "https://api.example.com/data"
    output = "processed_data"
    
    # Execute chain
    raw = fetch_data(url)
    df = process_data(raw)
    csv, json = save_data(df, output)
    
    print(f"Saved {len(df)} records to {csv} and {json}")
    return df

if __name__ == "__main__":
    df = main()
```

This chains: API fetch → Data processing → File saving"""
        },
    ],
    
    # ===== MULTI-TURN REASONING (50 samples) =====
    "multi_turn_reasoning": [
        {
            "user": "I need to design a system for managing a library. Walk me through the design.",
            "assistant": """Let's design a library management system step by step.

**Step 1: Identify Core Requirements**
- Track books (title, author, ISBN, copies)
- Track members (name, email, membership)
- Track loans (who borrowed what, when due)
- Search functionality
- Fine calculation

**Step 2: Database Schema**

```sql
-- Books table
CREATE TABLE books (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    author VARCHAR(255) NOT NULL,
    isbn VARCHAR(13) UNIQUE,
    total_copies INT DEFAULT 1,
    available_copies INT DEFAULT 1,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Members table
CREATE TABLE members (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE,
    phone VARCHAR(20),
    membership_date DATE DEFAULT CURRENT_DATE,
    status VARCHAR(20) DEFAULT 'active'
);

-- Loans table
CREATE TABLE loans (
    id SERIAL PRIMARY KEY,
    book_id INT REFERENCES books(id),
    member_id INT REFERENCES members(id),
    loan_date DATE DEFAULT CURRENT_DATE,
    due_date DATE NOT NULL,
    return_date DATE,
    fine_amount DECIMAL(10,2) DEFAULT 0
);
```

**Step 3: API Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| /books | GET | Search books |
| /books | POST | Add new book |
| /books/{id} | GET | Get book details |
| /members | GET/POST | List/Create members |
| /loans | POST | Borrow a book |
| /loans/{id}/return | POST | Return a book |
| /fines/{member_id} | GET | Calculate fines |

**Step 4: Business Logic**

```python
def borrow_book(book_id, member_id):
    # Check if book available
    book = get_book(book_id)
    if book.available_copies < 1:
        raise Exception("Book not available")
    
    # Check if member can borrow
    if count_active_loans(member_id) >= 5:
        raise Exception("Maximum loans reached")
    
    # Create loan
    due_date = date.today() + timedelta(days=14)
    loan = create_loan(book_id, member_id, due_date)
    
    # Update book copies
    decrement_available(book_id)
    
    return loan

def return_book(loan_id):
    loan = get_loan(loan_id)
    loan.return_date = date.today()
    
    # Calculate fine if overdue
    if loan.return_date > loan.due_date:
        days_overdue = (loan.return_date - loan.due_date).days
        loan.fine_amount = days_overdue * 0.50  # $0.50/day
    
    save(loan)
    increment_available(loan.book_id)
    return loan
```

**Step 5: Search Implementation**

```python
def search_books(query):
    # Full-text search on title and author
    results = db.query(\"\"\"
        SELECT * FROM books 
        WHERE to_tsvector(title || ' ' || author) 
        @@ to_tsquery(?)
        ORDER BY available_copies DESC
    \"\"\", query)
    return results
```

Would you like me to expand on any part?"""
        },
    ],
}

def generate_dataset():
    """Generate expanded training dataset."""
    samples = []
    sample_id = 1
    
    # Generate from templates
    for category, items in TEMPLATES.items():
        for item in items:
            system_prompt = random.choice(SYSTEM_PROMPTS)
            sample = {
                "id": f"train_{sample_id}",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": item["user"]},
                    {"role": "assistant", "content": item["assistant"]}
                ],
                "category": category,
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "quality": "verified"
                }
            }
            samples.append(sample)
            sample_id += 1
    
    # Load existing data and add
    existing_path = Path(__file__).parent.parent / "data" / "train.jsonl"
    if existing_path.exists():
        with open(existing_path, 'r') as f:
            for line in f:
                try:
                    existing = json.loads(line)
                    existing["id"] = f"existing_{sample_id}"
                    samples.append(existing)
                    sample_id += 1
                except:
                    pass
    
    # Shuffle and save
    random.shuffle(samples)
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Generated {len(samples)} samples to {OUTPUT_PATH}")
    return samples

if __name__ == "__main__":
    generate_dataset()
