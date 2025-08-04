"""
Sequence Analysis Utilities

Tools for analyzing operation sequences and call chains.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from collections import defaultdict
from .models import TrackedOperation, OperationStatus


class SequenceAnalyzer:
    """Analyze sequences of tracked operations."""
    
    def __init__(self, operations: List[TrackedOperation]):
        self.operations = operations
        self._by_thread = self._group_by_thread(operations)
    
    def _group_by_thread(self, operations: List[TrackedOperation]) -> Dict[str, List[TrackedOperation]]:
        """Group operations by thread ID."""
        grouped = defaultdict(list)
        for op in operations:
            grouped[op.thread_id].append(op)
        return dict(grouped)
    
    def get_thread_sequence(self, thread_id: str) -> List[TrackedOperation]:
        """Get operations for a specific thread in chronological order."""
        ops = self._by_thread.get(thread_id, [])
        return sorted(ops, key=lambda op: op.started_at)
    
    def get_call_tree(self, thread_id: str) -> Dict[str, Any]:
        """Build a hierarchical call tree for a thread."""
        ops = self.get_thread_sequence(thread_id)
        
        # Build parent-child relationships
        tree = {"root": []}
        op_map = {op.id: op for op in ops}
        
        for op in ops:
            parent_id = op.metadata.get("parent_operation_id")
            if parent_id and parent_id in op_map:
                if parent_id not in tree:
                    tree[parent_id] = []
                tree[parent_id].append(op)
            else:
                tree["root"].append(op)
        
        return self._build_tree_structure(tree, "root", op_map)
    
    def _build_tree_structure(self, tree: Dict, node_id: str, op_map: Dict) -> Dict:
        """Recursively build tree structure."""
        if node_id == "root":
            return {
                "operations": [
                    self._build_tree_structure(tree, op.id, op_map)
                    for op in tree.get("root", [])
                ]
            }
        
        op = op_map[node_id]
        return {
            "id": op.id,
            "name": op.operation_name,
            "status": op.status.value,
            "duration_ms": op.duration_ms,
            "started_at": op.started_at.isoformat(),
            "children": [
                self._build_tree_structure(tree, child.id, op_map)
                for child in tree.get(node_id, [])
            ]
        }
    
    def get_operation_flow(self, thread_id: str) -> List[Dict[str, Any]]:
        """Get a simplified flow of operations with timing."""
        ops = self.get_thread_sequence(thread_id)
        
        flow = []
        for i, op in enumerate(ops):
            flow_item = {
                "seq": i + 1,
                "name": op.operation_name,
                "status": op.status.value,
                "started_at": op.started_at.isoformat(),
                "duration_ms": op.duration_ms,
                "depth": op.metadata.get("depth", 0),
                "parent": op.metadata.get("parent_operation_id")
            }
            
            # Add time since previous operation
            if i > 0:
                time_since_prev = (op.started_at - ops[i-1].started_at).total_seconds() * 1000
                flow_item["ms_since_previous"] = time_since_prev
            
            flow.append(flow_item)
        
        return flow
    
    def get_call_chains(self, thread_id: str) -> List[List[str]]:
        """Get all unique call chains in a thread."""
        ops = self.get_thread_sequence(thread_id)
        
        chains = []
        for op in ops:
            chain = op.metadata.get("call_chain", [])
            chain.append(op.operation_name)
            if chain not in chains:
                chains.append(chain)
        
        return chains
    
    def get_statistics(self, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about operations."""
        if thread_id:
            ops = self.get_thread_sequence(thread_id)
        else:
            ops = self.operations
        
        if not ops:
            return {}
        
        # Group by operation name
        by_name = defaultdict(list)
        for op in ops:
            by_name[op.operation_name].append(op)
        
        stats = {
            "total_operations": len(ops),
            "unique_operations": len(by_name),
            "by_status": {},
            "by_operation": {},
            "timeline": {
                "start": min(op.started_at for op in ops).isoformat(),
                "end": max(op.started_at for op in ops).isoformat()
            }
        }
        
        # Status breakdown
        for status in OperationStatus:
            count = sum(1 for op in ops if op.status == status)
            if count > 0:
                stats["by_status"][status.value] = count
        
        # Per-operation statistics
        for name, name_ops in by_name.items():
            durations = [op.duration_ms for op in name_ops if op.duration_ms]
            stats["by_operation"][name] = {
                "count": len(name_ops),
                "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
                "min_duration_ms": min(durations) if durations else 0,
                "max_duration_ms": max(durations) if durations else 0,
                "success_rate": sum(1 for op in name_ops if op.status == OperationStatus.COMPLETED) / len(name_ops)
            }
        
        return stats
    
    def print_sequence(self, thread_id: str, show_timing: bool = True):
        """Print a formatted sequence of operations."""
        ops = self.get_thread_sequence(thread_id)
        
        print(f"\n📊 Operation Sequence for Thread: {thread_id}")
        print("=" * 80)
        
        for i, op in enumerate(ops):
            indent = "  " * op.metadata.get("depth", 0)
            status_icon = {
                OperationStatus.STARTED: "🔄",
                OperationStatus.COMPLETED: "✅",
                OperationStatus.FAILED: "❌",
                OperationStatus.INTERRUPTED: "⚠️"
            }.get(op.status, "❓")
            
            print(f"{indent}{status_icon} {op.operation_name}")
            
            if show_timing:
                timing_info = []
                if op.duration_ms:
                    timing_info.append(f"{op.duration_ms:.1f}ms")
                if i > 0:
                    gap = (op.started_at - ops[i-1].started_at).total_seconds() * 1000
                    timing_info.append(f"+{gap:.1f}ms")
                
                if timing_info:
                    print(f"{indent}   ⏱️  {' | '.join(timing_info)}")
        
        print("=" * 80)
        
        # Print summary
        if ops:
            total_time = (ops[-1].started_at - ops[0].started_at).total_seconds() * 1000
            print(f"\nSummary:")
            print(f"  Total operations: {len(ops)}")
            print(f"  Total time: {total_time:.1f}ms")
            print(f"  Success rate: {sum(1 for op in ops if op.status == OperationStatus.COMPLETED) / len(ops) * 100:.1f}%")


def analyze_sequences(operations: List[TrackedOperation]) -> SequenceAnalyzer:
    """Create a sequence analyzer for the given operations."""
    return SequenceAnalyzer(operations)