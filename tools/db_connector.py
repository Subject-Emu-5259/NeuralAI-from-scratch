# tools/db_connector.py
#
# Database connector for SQLite and PostgreSQL
# - Connect to databases
# - Execute queries
# - Schema inspection
# - Results as structured data

import sqlite3
import json
from typing import Dict, Any, List, Optional, Union
from pathlib import Path


class DatabaseConnector:
    """Connect to and query databases."""

    def __init__(self):
        self.connections: Dict[str, Any] = {}
        self.active_db: Optional[str] = None

    def connect_sqlite(self, path: str, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Connect to a SQLite database.
        
        Args:
            path: Path to SQLite file (or :memory: for in-memory)
            name: Optional name for the connection (defaults to path)
        
        Returns:
            {
                "success": bool,
                "name": str,
                "path": str,
                "tables": list
            }
        """
        conn_name = name or path
        
        try:
            # Create connection
            conn = sqlite3.connect(path)
            conn.row_factory = sqlite3.Row
            self.connections[conn_name] = conn
            self.active_db = conn_name
            
            # Get table list
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            return {
                "success": True,
                "name": conn_name,
                "path": path,
                "tables": tables
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "name": conn_name,
                "path": path,
                "tables": []
            }

    def close(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Close a database connection."""
        conn_name = name or self.active_db
        
        if not conn_name or conn_name not in self.connections:
            return {
                "success": False,
                "error": "Connection not found",
                "name": conn_name
            }
        
        try:
            self.connections[conn_name].close()
            del self.connections[conn_name]
            
            if self.active_db == conn_name:
                self.active_db = None
            
            return {
                "success": True,
                "name": conn_name
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "name": conn_name
            }

    def query(self, sql: str, params: Optional[tuple] = None, 
              name: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a SQL query.
        
        Args:
            sql: SQL query string
            params: Optional parameters for parameterized query
            name: Connection name (uses active_db if not specified)
        
        Returns:
            {
                "success": bool,
                "query": str,
                "rows": list of dicts,
                "row_count": int,
                "columns": list
            }
        """
        conn_name = name or self.active_db
        
        if not conn_name or conn_name not in self.connections:
            return {
                "success": False,
                "error": "No active database connection",
                "query": sql,
                "rows": [],
                "row_count": 0,
                "columns": []
            }
        
        try:
            conn = self.connections[conn_name]
            cursor = conn.cursor()
            
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            
            # Get column names
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            
            # Fetch results
            if sql.strip().upper().startswith(("SELECT", "PRAGMA", "SHOW", "DESCRIBE", "EXPLAIN")):
                rows = [dict(row) for row in cursor.fetchall()]
                row_count = len(rows)
            else:
                conn.commit()
                rows = []
                row_count = cursor.rowcount
            
            return {
                "success": True,
                "query": sql,
                "rows": rows,
                "row_count": row_count,
                "columns": columns
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": sql,
                "rows": [],
                "row_count": 0,
                "columns": []
            }

    def schema(self, table: Optional[str] = None, 
               name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get database schema information.
        
        Args:
            table: Optional specific table name (gets all if not specified)
            name: Connection name
        
        Returns:
            {
                "success": bool,
                "tables": [{"name": str, "columns": [{"name": str, "type": str, "nullable": bool}]}]
            }
        """
        conn_name = name or self.active_db
        
        if not conn_name or conn_name not in self.connections:
            return {
                "success": False,
                "error": "No active database connection",
                "tables": []
            }
        
        try:
            conn = self.connections[conn_name]
            cursor = conn.cursor()
            
            tables = []
            
            if table:
                # Get specific table schema
                cursor.execute(f"PRAGMA table_info({table})")
                columns = []
                for row in cursor.fetchall():
                    columns.append({
                        "name": row[1],
                        "type": row[2],
                        "nullable": not row[3],
                        "default": row[4],
                        "primary_key": bool(row[5])
                    })
                
                tables.append({
                    "name": table,
                    "columns": columns
                })
            else:
                # Get all tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                table_names = [row[0] for row in cursor.fetchall()]
                
                for table_name in table_names:
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = []
                    for row in cursor.fetchall():
                        columns.append({
                            "name": row[1],
                            "type": row[2],
                            "nullable": not row[3],
                            "default": row[4],
                            "primary_key": bool(row[5])
                        })
                    
                    tables.append({
                        "name": table_name,
                        "columns": columns
                    })
            
            return {
                "success": True,
                "tables": tables
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tables": []
            }

    def tables(self, name: Optional[str] = None) -> Dict[str, Any]:
        """List all tables in the database."""
        conn_name = name or self.active_db
        
        if not conn_name or conn_name not in self.connections:
            return {
                "success": False,
                "error": "No active database connection",
                "tables": []
            }
        
        try:
            conn = self.connections[conn_name]
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            return {
                "success": True,
                "tables": tables,
                "count": len(tables)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tables": []
            }

    def create_table(self, table: str, columns: List[Dict[str, str]], 
                    name: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new table.
        
        Args:
            table: Table name
            columns: List of {"name": str, "type": str, "primary_key": bool, "nullable": bool}
            name: Connection name
        
        Returns:
            {"success": bool, "table": str}
        """
        # Build column definitions
        col_defs = []
        for col in columns:
            col_def = f"{col['name']} {col.get('type', 'TEXT')}"
            if col.get('primary_key'):
                col_def += " PRIMARY KEY"
            if not col.get('nullable', True):
                col_def += " NOT NULL"
            if col.get('default'):
                col_def += f" DEFAULT {col['default']}"
            col_defs.append(col_def)
        
        sql = f"CREATE TABLE {table} ({', '.join(col_defs)})"
        
        return self.query(sql, name=name)

    def insert(self, table: str, data: Dict[str, Any], 
               name: Optional[str] = None) -> Dict[str, Any]:
        """
        Insert a row into a table.
        
        Args:
            table: Table name
            data: Column:value dict
            name: Connection name
        
        Returns:
            {"success": bool, "row_id": int}
        """
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        
        result = self.query(sql, params=tuple(data.values()), name=name)
        
        if result["success"]:
            conn_name = name or self.active_db
            conn = self.connections[conn_name]
            result["row_id"] = conn.cursor().lastrowid
        
        return result

    def update(self, table: str, data: Dict[str, Any], 
               where: str, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Update rows in a table.
        
        Args:
            table: Table name
            data: Column:value dict
            where: WHERE clause (without 'WHERE')
            name: Connection name
        
        Returns:
            {"success": bool, "row_count": int}
        """
        set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
        sql = f"UPDATE {table} SET {set_clause} WHERE {where}"
        
        return self.query(sql, params=tuple(data.values()), name=name)

    def delete(self, table: str, where: str, 
               name: Optional[str] = None) -> Dict[str, Any]:
        """
        Delete rows from a table.
        
        Args:
            table: Table name
            where: WHERE clause (without 'WHERE')
            name: Connection name
        
        Returns:
            {"success": bool, "row_count": int}
        """
        sql = f"DELETE FROM {table} WHERE {where}"
        return self.query(sql, name=name)


# Convenience functions
def connect(path: str, name: Optional[str] = None) -> DatabaseConnector:
    """Create a connector and connect to a SQLite database."""
    conn = DatabaseConnector()
    conn.connect_sqlite(path, name)
    return conn


if __name__ == "__main__":
    # Test with in-memory database
    db = DatabaseConnector()
    
    print("Creating in-memory database...")
    result = db.connect_sqlite(":memory:", "test")
    print(f"Connected: {result['success']}")
    
    print("\nCreating test table...")
    result = db.create_table("users", [
        {"name": "id", "type": "INTEGER", "primary_key": True},
        {"name": "name", "type": "TEXT", "nullable": False},
        {"name": "email", "type": "TEXT"}
    ])
    print(f"Table created: {result['success']}")
    
    print("\nInserting data...")
    result = db.insert("users", {"name": "Alice", "email": "alice@example.com"})
    print(f"Inserted row ID: {result.get('row_id')}")
    
    print("\nQuerying data...")
    result = db.query("SELECT * FROM users")
    print(f"Rows: {result['rows']}")
    
    print("\nSchema:")
    result = db.schema()
    for table in result['tables']:
        print(f"Table: {table['name']}")
        for col in table['columns']:
            print(f"  - {col['name']}: {col['type']}")
