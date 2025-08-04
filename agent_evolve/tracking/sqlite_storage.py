"""
SQLite Storage Implementation for Generic Tracker

Persistent storage using SQLite database.
"""

import sqlite3
import json
import asyncio
from typing import List, Optional, Dict
from datetime import datetime
from .models import TrackedMessage, TrackedOperation, OperationStatus


class SQLiteStorage:
    """SQLite storage implementation for tracking data."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = asyncio.Lock() if hasattr(asyncio, 'Lock') else None
        # Create database file immediately
        import os
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        # Initialize database synchronously
        self._initialize_sync()
    
    def _initialize_sync(self) -> None:
        """Initialize SQLite database synchronously."""
        try:
            import os
            resolved_path = os.path.abspath(self.db_path)
            print(f"🔧 Initializing SQLite database at: {self.db_path}")
            print(f"🔧 Resolved absolute path: {resolved_path}")
            conn = sqlite3.connect(self.db_path)
            try:
                # Create messages table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS tracked_messages (
                        id TEXT PRIMARY KEY,
                        thread_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT,
                        timestamp TEXT NOT NULL,
                        user_id TEXT,
                        metadata TEXT,
                        parent_id TEXT
                    )
                ''')
                
                # Create operations table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS tracked_operations (
                        id TEXT PRIMARY KEY,
                        thread_id TEXT NOT NULL,
                        operation_name TEXT NOT NULL,
                        status TEXT NOT NULL,
                        started_at TEXT NOT NULL,
                        ended_at TEXT,
                        duration_ms REAL,
                        input_data TEXT,
                        output_data TEXT,
                        error TEXT,
                        metadata TEXT
                    )
                ''')
                
                # Create indexes for performance
                conn.execute('CREATE INDEX IF NOT EXISTS idx_messages_thread_id ON tracked_messages(thread_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON tracked_messages(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_operations_thread_id ON tracked_operations(thread_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_operations_name ON tracked_operations(operation_name)')
                
                conn.commit()
                print(f"✅ Successfully created tracker database: {self.db_path}")
            finally:
                conn.close()
        except Exception as e:
            print(f"❌ Failed to create tracker database: {e}")
            raise

    async def initialize(self) -> None:
        """Initialize SQLite database and create tables."""
        # Already initialized in constructor
        pass
    
    async def close(self) -> None:
        """Close storage (no cleanup needed for SQLite)."""
        pass
    
    def store_message(self, message: TrackedMessage) -> None:
        """Store a message in SQLite (sync version)."""
        self._store_message_sync(message)
    
    async def store_message_async(self, message: TrackedMessage) -> None:
        """Store a message in SQLite (async version)."""
        if self._lock:
            async with self._lock:
                self._store_message_sync(message)
        else:
            self._store_message_sync(message)
    
    def _store_message_sync(self, message: TrackedMessage) -> None:
        """Internal sync method for storing message."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('''
                INSERT OR REPLACE INTO tracked_messages 
                (id, thread_id, role, content, timestamp, user_id, metadata, parent_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                message.id,
                message.thread_id,
                message.role,
                json.dumps(message.content) if not isinstance(message.content, str) else message.content,
                message.timestamp.isoformat(),
                message.user_id,
                json.dumps(message.metadata),
                message.parent_id
            ))
            conn.commit()
        finally:
            conn.close()
    
    def store_operation(self, operation: TrackedOperation) -> None:
        """Store an operation in SQLite (sync version)."""
        self._store_operation_sync(operation)
    
    async def store_operation_async(self, operation: TrackedOperation) -> None:
        """Store an operation in SQLite (async version)."""
        if self._lock:
            async with self._lock:
                self._store_operation_sync(operation)
        else:
            self._store_operation_sync(operation)
    
    def _store_operation_sync(self, operation: TrackedOperation) -> None:
        """Internal sync method for storing operation."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('''
                INSERT OR REPLACE INTO tracked_operations
                (id, thread_id, operation_name, status, started_at, ended_at, duration_ms, 
                 input_data, output_data, error, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                operation.id,
                operation.thread_id,
                operation.operation_name,
                operation.status.value,
                operation.started_at.isoformat(),
                operation.ended_at.isoformat() if operation.ended_at else None,
                operation.duration_ms,
                json.dumps(operation.input_data) if operation.input_data else None,
                json.dumps(operation.output_data) if operation.output_data else None,
                operation.error,
                json.dumps(operation.metadata)
            ))
            conn.commit()
        finally:
            conn.close()
    
    async def get_messages(self, 
                          thread_id: Optional[str] = None,
                          user_id: Optional[str] = None,
                          limit: int = 100) -> List[TrackedMessage]:
        """Retrieve messages with filters."""
        if self._lock:
            async with self._lock:
                return await self._get_messages_sync(thread_id, user_id, limit)
        else:
            return await self._get_messages_sync(thread_id, user_id, limit)
    
    async def _get_messages_sync(self, thread_id, user_id, limit) -> List[TrackedMessage]:
        """Internal sync method for getting messages."""
        conn = sqlite3.connect(self.db_path)
        try:
            query = "SELECT * FROM tracked_messages WHERE 1=1"
            params = []
            
            if thread_id:
                query += " AND thread_id = ?"
                params.append(thread_id)
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            messages = []
            for row in rows:
                content = row[3]
                try:
                    content = json.loads(content) if content and content.startswith('{') else content
                except:
                    pass
                
                metadata = json.loads(row[6]) if row[6] else {}
                
                message = TrackedMessage(
                    id=row[0],
                    thread_id=row[1],
                    role=row[2],
                    content=content,
                    timestamp=datetime.fromisoformat(row[4]),
                    user_id=row[5],
                    metadata=metadata,
                    parent_id=row[7]
                )
                messages.append(message)
            
            return list(reversed(messages))  # Return in chronological order
        finally:
            conn.close()
    
    async def get_operations(self,
                            thread_id: Optional[str] = None,
                            operation_name: Optional[str] = None,
                            limit: int = 100) -> List[TrackedOperation]:
        """Retrieve operations with filters."""
        if self._lock:
            async with self._lock:
                return await self._get_operations_sync(thread_id, operation_name, limit)
        else:
            return await self._get_operations_sync(thread_id, operation_name, limit)
    
    async def _get_operations_sync(self, thread_id, operation_name, limit) -> List[TrackedOperation]:
        """Internal sync method for getting operations."""
        conn = sqlite3.connect(self.db_path)
        try:
            query = "SELECT * FROM tracked_operations WHERE 1=1"
            params = []
            
            if thread_id:
                query += " AND thread_id = ?"
                params.append(thread_id)
            
            if operation_name:
                query += " AND operation_name = ?"
                params.append(operation_name)
            
            query += " ORDER BY started_at DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            operations = []
            for row in rows:
                input_data = json.loads(row[7]) if row[7] else None
                output_data = json.loads(row[8]) if row[8] else None
                metadata = json.loads(row[10]) if row[10] else {}
                
                operation = TrackedOperation(
                    id=row[0],
                    thread_id=row[1],
                    operation_name=row[2],
                    status=OperationStatus(row[3]),
                    started_at=datetime.fromisoformat(row[4]),
                    ended_at=datetime.fromisoformat(row[5]) if row[5] else None,
                    duration_ms=row[6],
                    input_data=input_data,
                    output_data=output_data,
                    error=row[9],
                    metadata=metadata
                )
                operations.append(operation)
            
            return list(reversed(operations))  # Return in chronological order
        finally:
            conn.close()
    
    async def batch_store_messages(self, messages: List[TrackedMessage]) -> None:
        """Store multiple messages."""
        if self._lock:
            async with self._lock:
                for message in messages:
                    await self._store_message_sync(message)
        else:
            for message in messages:
                await self._store_message_sync(message)
    
    async def batch_store_operations(self, operations: List[TrackedOperation]) -> None:
        """Store multiple operations."""
        if self._lock:
            async with self._lock:
                for operation in operations:
                    await self._store_operation_sync(operation)
        else:
            for operation in operations:
                await self._store_operation_sync(operation)