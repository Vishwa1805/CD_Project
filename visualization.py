"""
Visualization and Output Formatting Module

This module provides visualization and formatting capabilities for the CLR parser,
including HTML table generation, DOT format output, and parsing trace formatting.
"""

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import html
import re


@dataclass
class VisualizationConfig:
    """Configuration options for visualization output."""
    table_css_classes: str = "parse-table"
    trace_css_classes: str = "parsing-trace"
    error_css_classes: str = "error-message"
    include_inline_styles: bool = True
    compact_mode: bool = False


class HTMLTableGenerator:
    """Generates HTML tables for CLR parsing tables."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
    
    def generate_action_goto_tables_html(self, 
                                       action_table: Dict[Tuple[int, str], str],
                                       goto_table: Dict[Tuple[int, str], int],
                                       terminals: Set[str],
                                       non_terminals: Set[str]) -> str:
        """
        Generate combined HTML table for ACTION and GOTO tables.
        
        Args:
            action_table: Dictionary mapping (state, terminal) to action string
            goto_table: Dictionary mapping (state, non_terminal) to target state
            terminals: Set of terminal symbols
            non_terminals: Set of non-terminal symbols
            
        Returns:
            HTML string containing the combined parsing table
        """
        # Get all states
        all_states = set()
        for (state, _), _ in action_table.items():
            all_states.add(state)
        for (state, _), _ in goto_table.items():
            all_states.add(state)
        
        if not all_states:
            return self._generate_empty_table_html("No parsing states found")
        
        # Sort symbols for consistent display
        sorted_terminals = sorted(terminals - {'e'})  # Remove epsilon
        sorted_non_terminals = sorted(non_terminals)
        sorted_states = sorted(all_states)
        
        # Replace $end with $ for display
        display_terminals = []
        for term in sorted_terminals:
            if term == '$end':
                display_terminals.append('$')
            else:
                display_terminals.append(term)
        
        # Generate HTML
        html_lines = []
        
        # Note: CSS styles are now provided by external stylesheet with grammar-table classes
        
        # Start table with grammar table classes and accessibility attributes
        html_lines.append(f'<table class="grammar-table {self.config.table_css_classes}" role="table" aria-label="CLR Parsing Table with ACTION and GOTO sections">')
        
        # Generate header
        html_lines.append(self._generate_table_header(display_terminals, sorted_non_terminals))
        
        # Generate body
        html_lines.append('<tbody>')
        for state in sorted_states:
            html_lines.append(self._generate_table_row(
                state, sorted_terminals, sorted_non_terminals, 
                action_table, goto_table, display_terminals
            ))
        html_lines.append('</tbody>')
        
        html_lines.append('</table>')
        
        return '\n'.join(html_lines)
    
    def generate_action_table_html(self, 
                                 action_table: Dict[Tuple[int, str], str],
                                 terminals: Set[str]) -> str:
        """
        Generate HTML table for ACTION table only.
        
        Args:
            action_table: Dictionary mapping (state, terminal) to action string
            terminals: Set of terminal symbols
            
        Returns:
            HTML string containing the ACTION table
        """
        # Get all states
        all_states = set()
        for (state, _), _ in action_table.items():
            all_states.add(state)
        
        if not all_states:
            return self._generate_empty_table_html("No action table entries found")
        
        sorted_terminals = sorted(terminals - {'e'})
        sorted_states = sorted(all_states)
        
        # Replace $end with $ for display
        display_terminals = []
        for term in sorted_terminals:
            if term == '$end':
                display_terminals.append('$')
            else:
                display_terminals.append(term)
        
        html_lines = []
        
        # Note: CSS styles are now provided by external stylesheet with grammar-table classes
        
        html_lines.append(f'<table class="grammar-table {self.config.table_css_classes}" role="table" aria-label="ACTION Table for CLR Parser">')
        html_lines.append('<thead>')
        html_lines.append('<tr>')
        html_lines.append('<th class="grammar-table-header grammar-table-header-primary" scope="col">State</th>')
        
        for display_term in display_terminals:
            html_lines.append(f'<th class="grammar-table-header" scope="col">{html.escape(display_term)}</th>')
        
        html_lines.append('</tr>')
        html_lines.append('</thead>')
        html_lines.append('<tbody>')
        
        for state in sorted_states:
            html_lines.append('<tr>')
            html_lines.append(f'<th class="grammar-table-cell grammar-table-cell-primary" scope="row">{state}</th>')
            
            for i, term in enumerate(sorted_terminals):
                action = action_table.get((state, term), '')
                formatted_action = self._format_action(action)
                html_lines.append(f'<td class="grammar-table-cell">{formatted_action}</td>')
            
            html_lines.append('</tr>')
        
        html_lines.append('</tbody>')
        html_lines.append('</table>')
        
        return '\n'.join(html_lines)
    
    def generate_goto_table_html(self, 
                               goto_table: Dict[Tuple[int, str], int],
                               non_terminals: Set[str]) -> str:
        """
        Generate HTML table for GOTO table only.
        
        Args:
            goto_table: Dictionary mapping (state, non_terminal) to target state
            non_terminals: Set of non-terminal symbols
            
        Returns:
            HTML string containing the GOTO table
        """
        # Get all states
        all_states = set()
        for (state, _), _ in goto_table.items():
            all_states.add(state)
        
        if not all_states:
            return self._generate_empty_table_html("No goto table entries found")
        
        sorted_non_terminals = sorted(non_terminals)
        sorted_states = sorted(all_states)
        
        html_lines = []
        
        # Note: CSS styles are now provided by external stylesheet with grammar-table classes
        
        html_lines.append(f'<table class="grammar-table {self.config.table_css_classes}" role="table" aria-label="GOTO Table for CLR Parser">')
        html_lines.append('<thead>')
        html_lines.append('<tr>')
        html_lines.append('<th class="grammar-table-header grammar-table-header-primary" scope="col">State</th>')
        
        for non_term in sorted_non_terminals:
            html_lines.append(f'<th class="grammar-table-header" scope="col">{html.escape(non_term)}</th>')
        
        html_lines.append('</tr>')
        html_lines.append('</thead>')
        html_lines.append('<tbody>')
        
        for state in sorted_states:
            html_lines.append('<tr>')
            html_lines.append(f'<th class="grammar-table-cell grammar-table-cell-primary" scope="row">{state}</th>')
            
            for non_term in sorted_non_terminals:
                target_state = goto_table.get((state, non_term), '')
                html_lines.append(f'<td class="grammar-table-cell">{target_state}</td>')
            
            html_lines.append('</tr>')
        
        html_lines.append('</tbody>')
        html_lines.append('</table>')
        
        return '\n'.join(html_lines)
    
    def _generate_table_header(self, display_terminals: List[str], 
                             sorted_non_terminals: List[str]) -> str:
        """Generate the table header with ACTION and GOTO sections."""
        lines = []
        lines.append('<thead>')
        
        # First header row with ACTION and GOTO spans
        lines.append('<tr>')
        lines.append('<th class="grammar-table-header grammar-table-header-primary" scope="col" rowspan="2">State</th>')
        if display_terminals:
            lines.append(f'<th class="grammar-table-header" scope="colgroup" colspan="{len(display_terminals)}">ACTION</th>')
        if sorted_non_terminals:
            lines.append(f'<th class="grammar-table-header" scope="colgroup" colspan="{len(sorted_non_terminals)}">GOTO</th>')
        lines.append('</tr>')
        
        # Second header row with individual symbols
        lines.append('<tr>')
        for display_term in display_terminals:
            lines.append(f'<th class="grammar-table-header" scope="col">{html.escape(display_term)}</th>')
        for non_term in sorted_non_terminals:
            lines.append(f'<th class="grammar-table-header" scope="col">{html.escape(non_term)}</th>')
        lines.append('</tr>')
        
        lines.append('</thead>')
        return '\n'.join(lines)
    
    def _generate_table_row(self, state: int, sorted_terminals: List[str],
                          sorted_non_terminals: List[str],
                          action_table: Dict[Tuple[int, str], str],
                          goto_table: Dict[Tuple[int, str], int],
                          display_terminals: List[str]) -> str:
        """Generate a single table row for the given state."""
        lines = []
        lines.append('<tr>')
        lines.append(f'<th class="grammar-table-cell grammar-table-cell-primary" scope="row">{state}</th>')
        
        # ACTION columns
        for term in sorted_terminals:
            action = action_table.get((state, term), '')
            formatted_action = self._format_action(action)
            lines.append(f'<td class="grammar-table-cell">{formatted_action}</td>')
        
        # GOTO columns
        for non_term in sorted_non_terminals:
            target_state = goto_table.get((state, non_term), '')
            lines.append(f'<td class="grammar-table-cell">{target_state}</td>')
        
        lines.append('</tr>')
        return '\n'.join(lines)
    
    def _format_action(self, action: str) -> str:
        """Format an action string for HTML display."""
        if not action:
            return ''
        
        # Handle conflicts
        if action.startswith('[CONFLICT]'):
            conflict_part = action[10:].strip()  # Remove "[CONFLICT] " prefix
            actions = [a.strip() for a in conflict_part.split(' | ')]
            formatted_actions = []
            for a in actions:
                formatted_actions.append(f'<span class="conflict-action">{html.escape(a)}</span>')
            return '<span class="grammar-action-conflict">' + ' / '.join(formatted_actions) + '</span>'
        
        # Regular actions
        action = html.escape(action)
        
        # Add CSS classes for different action types using new grammar action classes
        if action.startswith('shift'):
            return f'<span class="grammar-action-shift">{action}</span>'
        elif action.startswith('reduce'):
            return f'<span class="grammar-action-reduce">{action}</span>'
        elif action == 'accept':
            return f'<span class="grammar-action-accept">{action}</span>'
        else:
            return action
    
    def _generate_empty_table_html(self, message: str) -> str:
        """Generate HTML for an empty table with a message."""
        html_lines = []
        # Note: CSS styles are now provided by external stylesheet
        html_lines.append(f'<div class="{self.config.error_css_classes}">')
        html_lines.append(f'<p>{html.escape(message)}</p>')
        html_lines.append('</div>')
        return '\n'.join(html_lines)
    
    def _generate_table_styles(self) -> str:
        """Generate inline CSS styles for the table."""
        return """
<style>
.parse-table {
    border-collapse: collapse;
    width: 100%;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    background-color: #1f2937;
    color: #f9fafb;
    border-radius: 8px;
    overflow: hidden;
}

.parse-table th, .parse-table td {
    border: 1px solid #374151;
    padding: 8px 12px;
    text-align: center;
}

.parse-table th {
    background-color: #111827;
    font-weight: bold;
    color: #60a5fa;
    text-transform: uppercase;
    font-size: 11px;
}

.parse-table td:first-child {
    background-color: #1f2937;
    font-weight: bold;
    text-align: left;
}

.parse-table tr:hover td {
    background-color: #374151;
}

/* Action type styling */
.shift-action {
    background-color: #1e40af;
    color: #dbeafe;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: bold;
    display: inline-block;
}

.reduce-action {
    background-color: #dc2626;
    color: #fecaca;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: bold;
    display: inline-block;
}

.accept-action {
    background-color: #16a34a;
    color: #dcfce7;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: bold;
    display: inline-block;
}

.conflict {
    background-color: #dc2626;
    color: #fecaca;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: bold;
    display: inline-block;
}

.conflict-action {
    font-size: 10px;
}

/* GOTO styling */
.parse-table td:not(:first-child):not(:has(.shift-action)):not(:has(.reduce-action)):not(:has(.accept-action)):not(:has(.conflict)) {
    color: #a78bfa;
    font-weight: bold;
}

.conflict-action {
    text-decoration: underline;
}

.error-message {
    color: #cc0000;
    font-weight: bold;
    padding: 10px;
    background-color: #ffeeee;
    border: 1px solid #cc0000;
    border-radius: 4px;
}
</style>
"""


class DOTGenerator:
    """Generates DOT format output for parse trees."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.node_counter = 0
    
    def generate_parse_tree_dot(self, parse_tree, title: str = "Parse Tree") -> str:
        """
        Generate compact DOT format representation of a parse tree.
        
        Args:
            parse_tree: ParseTreeNode object representing the root of the tree
            title: Title for the graph
            
        Returns:
            DOT format string
        """
        if not parse_tree:
            return self._generate_empty_tree_dot(title, "Parse tree is empty")
        
        self.node_counter = 0
        lines = []
        
        # Graph header with compact styling
        lines.append(f'digraph "{self._escape_dot_string(title)}" {{')
        lines.append('  rankdir=TB;')
        lines.append('  node [fontname="Arial", fontsize=12];')
        lines.append('  edge [fontsize=9, color="#333333"];')
        lines.append('  bgcolor=white;')
        lines.append('  splines=false;')
        lines.append('  nodesep=0.4;')
        lines.append('  ranksep=0.6;')
        lines.append('  margin=0.1;')
        lines.append('  size="8,6";')
        lines.append('  ratio=compress;')
        
        # Generate nodes and edges
        try:
            dot_content, _ = self._generate_node_dot(parse_tree)
            lines.append(dot_content)
        except Exception as e:
            lines.append(f'  error [label="Error generating tree: {self._escape_dot_string(str(e))}", shape=box, color=red];')
        
        lines.append('}')
        
        return '\n'.join(lines)
    
    def generate_automaton_dot(self, automaton, title: str = "CLR Automaton") -> str:
        """
        Generate DOT format representation of a CLR automaton.
        
        Args:
            automaton: CLRAutomaton object
            title: Title for the graph
            
        Returns:
            DOT format string
        """
        lines = []
        
        # Graph header
        lines.append(f'digraph "{self._escape_dot_string(title)}" {{')
        lines.append('  rankdir=LR;')
        lines.append('  node [shape=circle, fontname="Arial", fontsize=8];')
        lines.append('  edge [fontname="Arial", fontsize=8];')
        
        # Add states
        for state in automaton.states:
            state_label = self._format_state_label(state)
            if state.state_id == automaton.start_state_id:
                lines.append(f'  state{state.state_id} [label="{state_label}", style=bold];')
            else:
                lines.append(f'  state{state.state_id} [label="{state_label}"];')
        
        # Add transitions
        for (from_state, symbol), to_state in automaton.transitions.items():
            escaped_symbol = self._escape_dot_string(symbol)
            lines.append(f'  state{from_state} -> state{to_state} [label="{escaped_symbol}"];')
        
        lines.append('}')
        
        return '\n'.join(lines)
    
    def _generate_node_dot(self, node) -> Tuple[str, int]:
        """
        Generate compact DOT representation for a single node and its children.
        
        Returns:
            Tuple of (dot_string, next_node_id)
        """
        lines = []
        current_id = self.node_counter
        self.node_counter += 1
        
        # Create compact node styling
        escaped_label = self._escape_dot_string(node.label)
        
        if node.is_terminal:
            # Terminal nodes: compact blue boxes
            lines.append(f'  node{current_id} [label="{escaped_label}", shape=box, style=filled, fillcolor="#e3f2fd", color="#1976d2", fontname="Courier New", fontsize=11, width=0.6, height=0.4, margin=0.05];')
        else:
            # Non-terminal nodes: compact green circles
            lines.append(f'  node{current_id} [label="{escaped_label}", shape=circle, style=filled, fillcolor="#e8f5e8", color="#388e3c", fontname="Arial", fontsize=12, width=0.7, height=0.7, margin=0.05];')
        
        # Create edges to children
        for child in node.children:
            child_id = self.node_counter  # Pre-calculate child ID
            child_dot, _ = self._generate_node_dot(child)
            lines.append(child_dot)
            lines.append(f'  node{current_id} -> node{child_id} [color="#666666", penwidth=1.0];')
        
        return '\n'.join(lines), self.node_counter
    
    def _format_state_label(self, state) -> str:
        """Format a CLR state for DOT display."""
        if self.config.compact_mode:
            return str(state.state_id)
        
        # Show first few items
        items_text = []
        for i, item in enumerate(sorted(state.items, key=str)):
            if i >= 3:  # Limit to first 3 items
                items_text.append("...")
                break
            items_text.append(str(item))
        
        label = f"State {state.state_id}\\n" + "\\n".join(items_text)
        return self._escape_dot_string(label)
    
    def _generate_empty_tree_dot(self, title: str, message: str) -> str:
        """Generate DOT for an empty or error tree."""
        lines = []
        lines.append(f'digraph "{self._escape_dot_string(title)}" {{')
        lines.append('  rankdir=TB;')
        lines.append('  node [fontname="Arial"];')
        lines.append(f'  empty [label="{self._escape_dot_string(message)}", shape=box, color=red];')
        lines.append('}')
        return '\n'.join(lines)
    
    def _escape_dot_string(self, text: str) -> str:
        """Escape a string for use in DOT format."""
        if not text:
            return ""
        
        # Replace problematic characters
        text = str(text)
        text = text.replace('\\', '\\\\')
        text = text.replace('"', '\\"')
        text = text.replace('\n', '\\n')
        text = text.replace('\t', '\\t')
        text = text.replace('\r', '\\r')
        
        return text


class ParseTraceFormatter:
    """Formats parsing traces as HTML with step-by-step details."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
    
    def generate_trace_html(self, trace_steps: List, title: str = "Parsing Trace") -> str:
        """
        Generate HTML representation of parsing trace.
        
        Args:
            trace_steps: List of ParseStep objects
            title: Title for the trace
            
        Returns:
            HTML string showing step-by-step parsing
        """
        if not trace_steps:
            return self._generate_empty_trace_html("No parsing steps recorded")
        
        html_lines = []
        
        # Note: CSS styles are now provided by external stylesheet with grammar-table classes
        
        # Start trace container
        html_lines.append(f'<div class="{self.config.trace_css_classes}">')
        html_lines.append(f'<h3>{html.escape(title)}</h3>')
        
        # Create table using grammar table classes with accessibility
        html_lines.append('<table class="grammar-table trace-table" role="table" aria-label="Step-by-step parsing trace">')
        html_lines.append('<thead>')
        html_lines.append('<tr>')
        html_lines.append('<th class="grammar-table-header grammar-table-header-primary" scope="col">Step</th>')
        html_lines.append('<th class="grammar-table-header" scope="col">Stack</th>')
        html_lines.append('<th class="grammar-table-header" scope="col">Input</th>')
        html_lines.append('<th class="grammar-table-header" scope="col">Action</th>')
        html_lines.append('<th class="grammar-table-header" scope="col">Production</th>')
        html_lines.append('</tr>')
        html_lines.append('</thead>')
        html_lines.append('<tbody>')
        
        # Add trace steps
        for step in trace_steps:
            html_lines.append(self._format_trace_step(step))
        
        html_lines.append('</tbody>')
        html_lines.append('</table>')
        html_lines.append('</div>')
        
        return '\n'.join(html_lines)
    
    def generate_compact_trace_html(self, trace_steps: List) -> str:
        """
        Generate compact HTML representation of parsing trace.
        
        Args:
            trace_steps: List of ParseStep objects
            
        Returns:
            Compact HTML string showing parsing steps
        """
        if not trace_steps:
            return '<div class="trace-empty">No parsing steps</div>'
        
        html_lines = []
        html_lines.append('<div class="trace-compact">')
        
        for step in trace_steps:
            action_class = self._get_action_css_class(step.action)
            html_lines.append(f'<div class="trace-step {action_class}">')
            html_lines.append(f'<span class="step-number">{step.step_number}:</span> ')
            html_lines.append(f'<span class="step-action">{html.escape(str(step.action))}</span>')
            if step.production_used:
                html_lines.append(f' <span class="step-production">({html.escape(str(step.production_used))})</span>')
            html_lines.append('</div>')
        
        html_lines.append('</div>')
        
        return '\n'.join(html_lines)
    
    def _format_trace_step(self, step) -> str:
        """Format a single parsing step as HTML table row."""
        lines = []
        
        # Determine CSS class based on action type
        action_class = self._get_action_css_class(step.action)
        
        lines.append(f'<tr class="{action_class}">')
        
        # Step number
        lines.append(f'<td class="grammar-table-cell grammar-table-cell-primary step-number">{step.step_number}</td>')
        
        # Stack - use concatenated symbol_stack if available
        if hasattr(step, 'symbol_stack') and step.symbol_stack:
            stack_display = html.escape(step.symbol_stack)
        else:
            # Fallback to state-only stack
            stack_display = html.escape(' '.join(map(str, step.stack)))
        
        # Truncate if too long
        if len(stack_display) > 60:
            stack_display = stack_display[:57] + "..."
        
        lines.append(f'<td class="grammar-table-cell stack">{stack_display}</td>')
        
        # Input buffer
        input_tokens = []
        for i, token in enumerate(step.input_buffer):
            if i >= 10:  # Limit display
                input_tokens.append("...")
                break
            input_tokens.append(token.type if hasattr(token, 'type') else str(token))
        
        input_str = ' '.join(input_tokens)
        lines.append(f'<td class="grammar-table-cell input">{html.escape(input_str)}</td>')
        
        # Action with better formatting
        action_str = html.escape(str(step.action))
        
        # Add styling based on action type using new grammar action classes
        if 'shift' in action_str.lower():
            action_formatted = f'<span class="grammar-action-shift">{action_str}</span>'
        elif 'reduce' in action_str.lower():
            action_formatted = f'<span class="grammar-action-reduce">{action_str}</span>'
        elif 'accept' in action_str.lower():
            action_formatted = f'<span class="grammar-action-accept">{action_str}</span>'
        else:
            action_formatted = action_str
            
        lines.append(f'<td class="grammar-table-cell action">{action_formatted}</td>')
        
        # Production (if any)
        production_str = ""
        if step.production_used:
            production_str = html.escape(str(step.production_used))
        lines.append(f'<td class="grammar-table-cell production">{production_str}</td>')
        
        lines.append('</tr>')
        
        return '\n'.join(lines)
    
    def _get_action_css_class(self, action) -> str:
        """Get CSS class name for an action type."""
        action_str = str(action).lower()
        
        if 'shift' in action_str:
            return 'shift-step'
        elif 'reduce' in action_str:
            return 'reduce-step'
        elif 'accept' in action_str:
            return 'accept-step'
        else:
            return 'error-step'
    
    def _generate_empty_trace_html(self, message: str) -> str:
        """Generate HTML for empty trace."""
        html_lines = []
        # Note: CSS styles are now provided by external stylesheet
        html_lines.append(f'<div class="{self.config.error_css_classes}">')
        html_lines.append(f'<p>{html.escape(message)}</p>')
        html_lines.append('</div>')
        return '\n'.join(html_lines)
    
    def _generate_trace_styles(self) -> str:
        """Generate inline CSS styles for the trace with improved contrast."""
        return """
<style>
.parsing-trace {
    font-family: 'Courier New', monospace;
    font-size: 14px;
    background-color: #1f2937;
    color: #f9fafb;
    border-radius: 8px;
    overflow: hidden;
}

.trace-table {
    border-collapse: collapse;
    width: 100%;
    margin-top: 10px;
    background-color: #1f2937;
    color: #f9fafb;
}

.trace-table th, .trace-table td {
    border: 1px solid #374151;
    padding: 8px 12px;
    text-align: left;
    font-size: 13px;
}

.trace-table th {
    background-color: #111827;
    font-weight: 700;
    color: #60a5fa;
    text-transform: uppercase;
    font-size: 12px;
}

.trace-table td:first-child {
    background-color: #111827;
    font-weight: 700;
    text-align: center;
    color: #60a5fa;
}

.shift-step {
    background-color: #1e3a8a;
}

.shift-step td {
    color: #dbeafe;
}

.reduce-step {
    background-color: #7c2d12;
}

.reduce-step td {
    color: #fed7aa;
}

.accept-step {
    background-color: #14532d;
}

.accept-step td {
    color: #bbf7d0;
}

.error-step {
    background-color: #7f1d1d;
}

.error-step td {
    color: #fecaca;
}

.step-number {
    font-weight: 700;
    text-align: center;
}

.stack {
    font-family: 'Courier New', monospace;
    background-color: #374151;
    color: #f3f4f6;
    font-weight: 500;
}

.input {
    font-family: 'Courier New', monospace;
    background-color: #1e40af;
    color: #dbeafe;
    font-weight: 500;
}

.action {
    font-weight: 700;
    color: #ffffff;
    text-align: center;
}

.production {
    font-style: italic;
    color: #d1d5db;
    font-weight: 500;
}

.trace-compact {
    font-family: 'Courier New', monospace;
    font-size: 13px;
    background-color: #1f2937;
    padding: 10px;
    border-radius: 6px;
}

.trace-step {
    margin: 4px 0;
    padding: 4px 8px;
    border-radius: 4px;
    color: #f9fafb;
}

.trace-empty {
    color: #9ca3af;
    font-style: italic;
    padding: 20px;
    text-align: center;
    background-color: #1f2937;
}

/* Action type styling in trace */
.trace-table .shift-action {
    background-color: #3b82f6 !important;
    color: #ffffff !important;
    padding: 2px 6px;
    border-radius: 3px;
    font-weight: 700;
    font-size: 12px;
    display: inline-block;
}

.trace-table .reduce-action {
    background-color: #ef4444 !important;
    color: #ffffff !important;
    padding: 2px 6px;
    border-radius: 3px;
    font-weight: 700;
    font-size: 12px;
    display: inline-block;
}

.trace-table .accept-action {
    background-color: #22c55e !important;
    color: #ffffff !important;
    padding: 2px 6px;
    border-radius: 3px;
    font-weight: 700;
    font-size: 12px;
    display: inline-block;
}
</style>
"""


class ErrorMessageFormatter:
    """Formats error messages with proper styling and context."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
    
    def format_parse_error(self, error_message: str, error_position: int = -1, 
                          input_string: str = "", context_length: int = 20) -> str:
        """
        Format a parsing error message with context.
        
        Args:
            error_message: The error message
            error_position: Position in input where error occurred
            input_string: The input string being parsed
            context_length: Number of characters to show around error position
            
        Returns:
            Formatted HTML error message
        """
        html_lines = []
        
        if self.config.include_inline_styles:
            html_lines.append(self._generate_error_styles())
        
        html_lines.append(f'<div class="{self.config.error_css_classes}">')
        html_lines.append('<h4>Parse Error</h4>')
        html_lines.append(f'<p class="error-text">{html.escape(error_message)}</p>')
        
        # Add context if position is provided
        if error_position >= 0 and input_string:
            context_html = self._generate_error_context(input_string, error_position, context_length)
            html_lines.append(context_html)
        
        html_lines.append('</div>')
        
        return '\n'.join(html_lines)
    
    def format_conflict_report(self, conflicts: List) -> str:
        """
        Format a conflict report as HTML.
        
        Args:
            conflicts: List of Conflict objects
            
        Returns:
            Formatted HTML conflict report
        """
        if not conflicts:
            return '<div class="no-conflicts">No conflicts detected in the grammar.</div>'
        
        html_lines = []
        
        if self.config.include_inline_styles:
            html_lines.append(self._generate_error_styles())
        
        html_lines.append('<div class="conflict-report">')
        html_lines.append(f'<h4>Grammar Conflicts ({len(conflicts)} found)</h4>')
        
        for i, conflict in enumerate(conflicts, 1):
            html_lines.append(f'<div class="conflict-item">')
            html_lines.append(f'<h5>Conflict {i}: {html.escape(conflict.conflict_type)}</h5>')
            html_lines.append(f'<p><strong>State:</strong> {conflict.state_id}</p>')
            html_lines.append(f'<p><strong>Symbol:</strong> {html.escape(conflict.symbol)}</p>')
            html_lines.append(f'<p><strong>Description:</strong> {html.escape(conflict.description)}</p>')
            
            if conflict.actions:
                html_lines.append('<p><strong>Conflicting Actions:</strong></p>')
                html_lines.append('<ul>')
                for action in conflict.actions:
                    html_lines.append(f'<li>{html.escape(action)}</li>')
                html_lines.append('</ul>')
            
            html_lines.append('</div>')
        
        html_lines.append('</div>')
        
        return '\n'.join(html_lines)
    
    def format_lexical_errors(self, errors: List) -> str:
        """
        Format lexical analysis errors as HTML.
        
        Args:
            errors: List of LexicalError objects
            
        Returns:
            Formatted HTML error report
        """
        if not errors:
            return '<div class="no-errors">No lexical errors found.</div>'
        
        html_lines = []
        
        if self.config.include_inline_styles:
            html_lines.append(self._generate_error_styles())
        
        html_lines.append('<div class="lexical-errors">')
        html_lines.append(f'<h4>Lexical Errors ({len(errors)} found)</h4>')
        
        for i, error in enumerate(errors, 1):
            html_lines.append('<div class="error-item">')
            html_lines.append(f'<h5>Error {i}</h5>')
            html_lines.append(f'<p><strong>Message:</strong> {html.escape(error.message)}</p>')
            html_lines.append(f'<p><strong>Position:</strong> Line {error.line}, Column {error.column}</p>')
            
            if hasattr(error, 'context') and error.context:
                html_lines.append(f'<p><strong>Context:</strong> <code>{html.escape(error.context)}</code></p>')
            
            html_lines.append('</div>')
        
        html_lines.append('</div>')
        
        return '\n'.join(html_lines)
    
    def _generate_error_context(self, input_string: str, error_position: int, 
                              context_length: int) -> str:
        """Generate HTML showing error context in input string."""
        start_pos = max(0, error_position - context_length)
        end_pos = min(len(input_string), error_position + context_length)
        
        before_error = input_string[start_pos:error_position]
        error_char = input_string[error_position] if error_position < len(input_string) else ''
        after_error = input_string[error_position + 1:end_pos]
        
        html_lines = []
        html_lines.append('<div class="error-context">')
        html_lines.append('<p><strong>Context:</strong></p>')
        html_lines.append('<pre class="context-display">')
        
        if start_pos > 0:
            html_lines.append('...')
        
        html_lines.append(html.escape(before_error))
        html_lines.append(f'<span class="error-position">{html.escape(error_char)}</span>')
        html_lines.append(html.escape(after_error))
        
        if end_pos < len(input_string):
            html_lines.append('...')
        
        html_lines.append('</pre>')
        html_lines.append(f'<p class="position-info">Error at position {error_position}</p>')
        html_lines.append('</div>')
        
        return '\n'.join(html_lines)
    
    def _generate_error_styles(self) -> str:
        """Generate inline CSS styles for error messages."""
        return """
<style>
.error-message {
    color: #cc0000;
    background-color: #ffeeee;
    border: 1px solid #cc0000;
    border-radius: 4px;
    padding: 10px;
    margin: 10px 0;
    font-family: Arial, sans-serif;
}

.error-text {
    font-weight: bold;
    margin: 5px 0;
}

.conflict-report {
    background-color: #fff8e1;
    border: 1px solid #ff9800;
    border-radius: 4px;
    padding: 10px;
    margin: 10px 0;
}

.conflict-item {
    margin: 10px 0;
    padding: 8px;
    background-color: #ffffff;
    border-left: 3px solid #ff9800;
}

.lexical-errors {
    background-color: #ffebee;
    border: 1px solid #f44336;
    border-radius: 4px;
    padding: 10px;
    margin: 10px 0;
}

.error-item {
    margin: 10px 0;
    padding: 8px;
    background-color: #ffffff;
    border-left: 3px solid #f44336;
}

.error-context {
    margin: 10px 0;
    padding: 8px;
    background-color: #f8f8f8;
    border-radius: 4px;
}

.context-display {
    font-family: monospace;
    background-color: #ffffff;
    padding: 8px;
    border: 1px solid #ddd;
    border-radius: 2px;
    overflow-x: auto;
}

.error-position {
    background-color: #ff0000;
    color: #ffffff;
    font-weight: bold;
    padding: 0 2px;
}

.position-info {
    font-size: 12px;
    color: #666;
    margin: 5px 0 0 0;
}

.no-conflicts, .no-errors {
    color: #4caf50;
    font-weight: bold;
    padding: 10px;
    background-color: #e8f5e8;
    border: 1px solid #4caf50;
    border-radius: 4px;
}
</style>
"""


class VisualizationGenerator:
    """Main visualization generator that combines all formatting capabilities."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.table_generator = HTMLTableGenerator(config)
        self.dot_generator = DOTGenerator(config)
        self.trace_formatter = ParseTraceFormatter(config)
        self.error_formatter = ErrorMessageFormatter(config)
    
    def generate_complete_visualization(self, 
                                     parsing_tables=None,
                                     parse_tree=None,
                                     trace_steps=None,
                                     conflicts=None,
                                     lexical_errors=None,
                                     terminals=None,
                                     non_terminals=None) -> Dict[str, str]:
        """
        Generate complete visualization output for all components.
        
        Args:
            parsing_tables: ParsingTables object with action and goto tables
            parse_tree: ParseTreeNode object
            trace_steps: List of ParseStep objects
            conflicts: List of Conflict objects
            lexical_errors: List of LexicalError objects
            terminals: Set of terminal symbols
            non_terminals: Set of non-terminal symbols
            
        Returns:
            Dictionary with keys: 'tables_html', 'tree_dot', 'trace_html', 
            'conflicts_html', 'errors_html'
        """
        result = {}
        
        # Generate parsing tables HTML
        if parsing_tables and terminals and non_terminals:
            result['tables_html'] = self.table_generator.generate_action_goto_tables_html(
                parsing_tables.action_table,
                parsing_tables.goto_table,
                terminals,
                non_terminals
            )
        else:
            result['tables_html'] = self.table_generator._generate_empty_table_html(
                "No parsing tables available"
            )
        
        # Generate parse tree DOT
        if parse_tree:
            result['tree_dot'] = self.dot_generator.generate_parse_tree_dot(parse_tree)
        else:
            result['tree_dot'] = self.dot_generator._generate_empty_tree_dot(
                "Parse Tree", "No parse tree available"
            )
        
        # Generate trace HTML
        if trace_steps:
            result['trace_html'] = self.trace_formatter.generate_trace_html(trace_steps)
        else:
            result['trace_html'] = self.trace_formatter._generate_empty_trace_html(
                "No parsing trace available"
            )
        
        # Generate conflicts HTML
        if conflicts is not None:
            result['conflicts_html'] = self.error_formatter.format_conflict_report(conflicts)
        else:
            result['conflicts_html'] = '<div class="no-conflicts">No conflict analysis performed.</div>'
        
        # Generate lexical errors HTML
        if lexical_errors is not None:
            result['errors_html'] = self.error_formatter.format_lexical_errors(lexical_errors)
        else:
            result['errors_html'] = '<div class="no-errors">No lexical analysis performed.</div>'
        
        return result
    
    def generate_parsing_tables_html(self, parsing_tables, terminals: Set[str], 
                                   non_terminals: Set[str]) -> str:
        """Generate HTML for parsing tables."""
        return self.table_generator.generate_action_goto_tables_html(
            parsing_tables.action_table,
            parsing_tables.goto_table,
            terminals,
            non_terminals
        )
    
    def generate_parse_tree_dot(self, parse_tree, title: str = "Parse Tree") -> str:
        """Generate DOT format for parse tree."""
        return self.dot_generator.generate_parse_tree_dot(parse_tree, title)
    
    def generate_trace_html(self, trace_steps: List, title: str = "Parsing Trace") -> str:
        """Generate HTML for parsing trace."""
        return self.trace_formatter.generate_trace_html(trace_steps, title)
    
    def format_error_message(self, error_message: str, error_position: int = -1, 
                           input_string: str = "") -> str:
        """Format an error message."""
        return self.error_formatter.format_parse_error(
            error_message, error_position, input_string
        )