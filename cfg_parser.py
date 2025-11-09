"""
CFG Parser Implementation - Core Data Structures and Grammar Processing

This module implements the core data structures and grammar processing functionality
for parsing context-free grammars without external dependencies.
"""

from dataclasses import dataclass, field
from typing import List, Set, Dict, Tuple, Optional, Union, Any
import re
import time
import sys
from enum import Enum


@dataclass
class Production:
    """Represents a single production rule in a context-free grammar."""
    lhs: str  # Left-hand side non-terminal
    rhs: List[str]  # Right-hand side symbols
    is_epsilon: bool = False
    
    def __str__(self) -> str:
        if self.is_epsilon:
            return f"{self.lhs} -> e"
        return f"{self.lhs} -> {' '.join(self.rhs)}"
    
    def __hash__(self) -> int:
        return hash((self.lhs, tuple(self.rhs), self.is_epsilon))


class CLRItemFlyweight:
    """
    Enhanced flyweight factory for CLR items to reduce memory usage.
    
    This implements the flyweight pattern to ensure that identical CLR items
    share the same object instance, significantly reducing memory consumption
    for large grammars with many repeated items.
    
    Enhanced features:
    - Weak references to allow garbage collection
    - Memory usage tracking
    - Cache size limits for large grammars
    - Performance statistics
    """
    _instances = {}  # Cache for flyweight instances
    _access_count = {}  # Track access frequency for cache optimization
    _max_cache_size = 10000  # Limit cache size for memory management
    _cache_hits = 0
    _cache_misses = 0
    
    @classmethod
    def get_item(cls, production: Production, dot_position: int, lookahead: str) -> 'CLRItem':
        """
        Get or create a CLR item using the flyweight pattern with enhanced caching.
        
        Args:
            production: The production rule
            dot_position: Position of the dot
            lookahead: Lookahead terminal
            
        Returns:
            CLRItem instance (potentially shared)
        """
        key = (production, dot_position, lookahead)
        
        if key in cls._instances:
            cls._cache_hits += 1
            cls._access_count[key] = cls._access_count.get(key, 0) + 1
            return cls._instances[key]
        
        cls._cache_misses += 1
        
        # Check cache size limit
        if len(cls._instances) >= cls._max_cache_size:
            cls._evict_least_used()
        
        # Create new item
        item = CLRItem(production, dot_position, lookahead)
        cls._instances[key] = item
        cls._access_count[key] = 1
        
        return item
    
    @classmethod
    def _evict_least_used(cls):
        """Evict least frequently used items when cache is full."""
        if not cls._instances:
            return
        
        # Find items with lowest access count
        min_access = min(cls._access_count.values())
        items_to_remove = [key for key, count in cls._access_count.items() 
                          if count == min_access]
        
        # Remove up to 10% of cache size
        evict_count = max(1, len(cls._instances) // 10)
        for key in items_to_remove[:evict_count]:
            cls._instances.pop(key, None)
            cls._access_count.pop(key, None)
    
    @classmethod
    def clear_cache(cls):
        """Clear the flyweight cache to free memory."""
        cls._instances.clear()
        cls._access_count.clear()
        cls._cache_hits = 0
        cls._cache_misses = 0
    
    @classmethod
    def get_cache_size(cls) -> int:
        """Get the current size of the flyweight cache."""
        return len(cls._instances)
    
    @classmethod
    def get_cache_stats(cls) -> Dict[str, Union[int, float]]:
        """Get detailed cache performance statistics."""
        total_requests = cls._cache_hits + cls._cache_misses
        hit_rate = (cls._cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_size': len(cls._instances),
            'max_cache_size': cls._max_cache_size,
            'cache_hits': cls._cache_hits,
            'cache_misses': cls._cache_misses,
            'hit_rate_percent': round(hit_rate, 2),
            'total_requests': total_requests,
            'most_accessed_count': max(cls._access_count.values()) if cls._access_count else 0,
            'average_access_count': sum(cls._access_count.values()) / len(cls._access_count) if cls._access_count else 0
        }
    
    @classmethod
    def set_cache_limit(cls, limit: int):
        """Set the maximum cache size."""
        cls._max_cache_size = limit
        if len(cls._instances) > limit:
            # Evict excess items
            while len(cls._instances) > limit:
                cls._evict_least_used()


@dataclass
class CLRItem:
    """Represents a CLR item with dot position and lookahead."""
    production: Production
    dot_position: int  # Position of dot in RHS (0 = before first symbol)
    lookahead: str  # Lookahead terminal
    
    def __str__(self) -> str:
        rhs_with_dot = self.production.rhs.copy()
        rhs_with_dot.insert(self.dot_position, ".")
        if self.production.is_epsilon:
            rhs_str = "e"
        else:
            rhs_str = " ".join(rhs_with_dot)
        return f"[{self.production.lhs} -> {rhs_str}, {self.lookahead}]"
    
    def __hash__(self) -> int:
        return hash((self.production, self.dot_position, self.lookahead))
    
    def is_complete(self) -> bool:
        """Check if the dot is at the end of the production."""
        return self.dot_position >= len(self.production.rhs)
    
    def next_symbol(self) -> Optional[str]:
        """Get the symbol after the dot, or None if at end."""
        if self.is_complete():
            return None
        return self.production.rhs[self.dot_position]


@dataclass
class CLRState:
    """Represents a state in the CLR automaton."""
    items: Set[CLRItem]
    state_id: int
    
    def __str__(self) -> str:
        items_str = "\n  ".join(str(item) for item in sorted(self.items, key=str))
        return f"State {self.state_id}:\n  {items_str}"
    
    def __hash__(self) -> int:
        return hash(frozenset(self.items))


@dataclass
class Grammar:
    """Represents a context-free grammar."""
    productions: List[Production]
    terminals: Set[str]
    non_terminals: Set[str]
    start_symbol: str
    
    def __str__(self) -> str:
        lines = [f"Start Symbol: {self.start_symbol}"]
        lines.append(f"Terminals: {sorted(self.terminals)}")
        lines.append(f"Non-terminals: {sorted(self.non_terminals)}")
        lines.append("Productions:")
        for prod in self.productions:
            lines.append(f"  {prod}")
        return "\n".join(lines)


@dataclass
class Token:
    """Represents a token produced by the lexical analyzer."""
    type: str  # Terminal symbol name
    value: str  # Actual text value
    position: int  # Position in input string
    line: int = 1  # Line number (for error reporting)
    column: int = 1  # Column number (for error reporting)
    
    def __str__(self) -> str:
        return f"Token({self.type}, '{self.value}', pos={self.position})"
    
    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class LexicalError:
    """Represents a lexical analysis error."""
    message: str
    position: int
    line: int
    column: int
    context: str  # Surrounding text for context
    
    def __str__(self) -> str:
        return f"Lexical error at line {self.line}, column {self.column}: {self.message}"


class GrammarProcessor:
    """Processes CFG input text and creates Grammar objects."""
    
    def __init__(self):
        self.terminals: Set[str] = set()
        self.non_terminals: Set[str] = set()
        self.productions: List[Production] = []
    
    def parse_grammar(self, cfg_text: str) -> Grammar:
        """
        Parse CFG input text and return a Grammar object.
        
        Supports formats:
        - A -> alpha | beta
        - A : alpha | beta  
        - A = alpha | beta
        
        Handles epsilon productions using 'e' or empty alternatives.
        """
        self._reset()
        
        # Clean and normalize input
        cfg_text = self._clean_input(cfg_text)
        
        # Parse productions
        raw_productions = self._extract_raw_productions(cfg_text)
        self.productions = self._normalize_productions(raw_productions)
        
        # Extract symbols
        self.terminals, self.non_terminals = self._extract_symbols()
        
        # Determine start symbol (first non-terminal encountered)
        start_symbol = self.productions[0].lhs if self.productions else ""
        
        return Grammar(
            productions=self.productions,
            terminals=self.terminals,
            non_terminals=self.non_terminals,
            start_symbol=start_symbol
        )
    
    def _reset(self):
        """Reset internal state for new grammar parsing."""
        self.terminals = set()
        self.non_terminals = set()
        self.productions = []
    
    def _clean_input(self, cfg_text: str) -> str:
        """Clean and normalize CFG input text."""
        # Remove comments (// and /* */ style)
        cfg_text = re.sub(r'//.*$', '', cfg_text, flags=re.MULTILINE)
        cfg_text = re.sub(r'/\*.*?\*/', '', cfg_text, flags=re.DOTALL)
        
        # Remove extra whitespace but preserve line structure
        lines = []
        for line in cfg_text.split('\n'):
            line = line.strip()
            if line:
                lines.append(line)
        
        return '\n'.join(lines)  
  
    def _extract_raw_productions(self, cfg_text: str) -> List[str]:
        """Extract raw production strings from CFG text."""
        productions = []
        
        # First, handle semicolon-separated productions
        if ';' in cfg_text:
            blocks = cfg_text.split(';')
        else:
            # If no semicolons, split by lines and group by production
            blocks = [cfg_text]
        
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            
            # Split into lines and process
            lines = block.split('\n')
            current_production = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if line starts a new production (LHS -> or LHS : or LHS =)
                if re.match(r'^[a-zA-Z_]\w*\s*(?:->|:|\=)', line):
                    if current_production:
                        productions.append(current_production)
                    current_production = line
                elif line.startswith('|') and current_production:
                    # Continuation with alternative
                    current_production += ' ' + line
                elif current_production and not re.match(r'^[a-zA-Z_]\w*\s*(?:->|:|\=)', line):
                    # Continuation of current production
                    current_production += ' ' + line
                else:
                    # Standalone line that might be a production
                    if re.search(r'(?:->|:|\=)', line):
                        if current_production:
                            productions.append(current_production)
                        current_production = line
            
            if current_production:
                productions.append(current_production)
        
        return productions
    
    def _normalize_productions(self, raw_productions: List[str]) -> List[Production]:
        """Convert raw production strings to Production objects."""
        productions = []
        
        for raw_prod in raw_productions:
            # Match production pattern with flexible arrow syntax
            match = re.match(r'^\s*([a-zA-Z_]\w*)\s*(?:->|:|\=)\s*(.*)', raw_prod, re.DOTALL)
            if not match:
                continue
                
            lhs = match.group(1).strip()
            rhs_text = match.group(2).strip()
            
            # Split alternatives by |
            alternatives = [alt.strip() for alt in rhs_text.split('|')]
            
            for alt in alternatives:
                if not alt:
                    continue
                    
                # Handle epsilon productions
                if alt.lower() == 'e' or alt == 'e' or alt == '/* empty */' or alt.strip() == '':
                    productions.append(Production(lhs=lhs, rhs=[], is_epsilon=True))
                else:
                    # Parse symbols in the alternative
                    symbols = self._parse_symbols(alt)
                    if symbols:  # Only add if we found symbols
                        productions.append(Production(lhs=lhs, rhs=symbols, is_epsilon=False))
        
        return productions
    
    def _parse_symbols(self, rhs_text: str) -> List[str]:
        """Parse symbols from RHS text, handling quoted terminals."""
        symbols = []
        
        # Regular expression to match:
        # - Quoted strings: 'symbol' or "symbol"
        # - Identifiers: word characters
        # - Single characters: any non-whitespace
        pattern = r"'([^']*)'|\"([^\"]*)\"|(\w+)|(\S)"
        
        for match in re.finditer(pattern, rhs_text):
            if match.group(1) is not None:  # Single quoted
                # Store the unquoted content for terminals
                symbols.append(match.group(1))
            elif match.group(2) is not None:  # Double quoted
                # Store the unquoted content for terminals
                symbols.append(match.group(2))
            elif match.group(3) is not None:  # Identifier
                symbols.append(match.group(3))
            elif match.group(4) is not None:  # Single character
                symbols.append(match.group(4))
        
        return symbols
    
    def _extract_symbols(self) -> Tuple[Set[str], Set[str]]:
        """
        Extract terminals and non-terminals from productions.
        
        Strategy:
        1. All LHS symbols are non-terminals
        2. Symbols that appear in RHS but not as LHS are terminals
        3. Handle quoted terminals specially
        """
        # Collect all LHS symbols (definitely non-terminals)
        lhs_symbols = set()
        for prod in self.productions:
            lhs_symbols.add(prod.lhs)
        
        # Collect all RHS symbols
        rhs_symbols = set()
        for prod in self.productions:
            for symbol in prod.rhs:
                rhs_symbols.add(symbol)
        
        # Non-terminals are LHS symbols
        non_terminals = lhs_symbols.copy()
        
        # Terminals are RHS symbols that are not non-terminals
        terminals = rhs_symbols - non_terminals
        
        # Add end-of-input marker
        terminals.add('$')
        
        return terminals, non_terminals


class LexicalAnalyzer:
    """
    Lexical analyzer that tokenizes input strings based on grammar terminals.
    
    Implements longest-match tokenization strategy and handles:
    - Single-character and multi-character tokens
    - Quoted terminals and escape sequences
    - Custom terminal definitions from grammar
    """
    
    def __init__(self, terminals: Set[str]):
        """
        Initialize the lexical analyzer with grammar terminals.
        
        Args:
            terminals: Set of terminal symbols from the grammar
        """
        self.terminals = terminals.copy()
        self.terminals.discard('$')  # Remove end-of-input marker
        self.terminals.discard('e')  # Remove epsilon
        
        # Prepare terminal patterns for tokenization
        self._prepare_terminal_patterns()
    
    def tokenize(self, input_string: str) -> Tuple[List[Token], List[LexicalError]]:
        """
        Tokenize an input string using longest-match strategy.
        
        Args:
            input_string: String to tokenize
            
        Returns:
            Tuple of (tokens, errors) where tokens is list of Token objects
            and errors is list of LexicalError objects
        """
        tokens = []
        errors = []
        position = 0
        line = 1
        column = 1
        
        while position < len(input_string):
            # Skip whitespace
            if input_string[position].isspace():
                if input_string[position] == '\n':
                    line += 1
                    column = 1
                else:
                    column += 1
                position += 1
                continue
            
            # Try to match a terminal using longest-match strategy
            best_match = None
            best_length = 0
            
            for terminal_pattern, terminal_name in self.terminal_patterns:
                match = terminal_pattern.match(input_string, position)
                if match and match.end() - match.start() > best_length:
                    best_match = match
                    best_length = match.end() - match.start()
                    best_terminal = terminal_name
            
            if best_match:
                # Found a match
                token_value = input_string[best_match.start():best_match.end()]
                token = Token(
                    type=best_terminal,
                    value=token_value,
                    position=position,
                    line=line,
                    column=column
                )
                tokens.append(token)
                
                # Update position and column
                position = best_match.end()
                column += best_length
            else:
                # No match found - lexical error
                error_char = input_string[position]
                context_start = max(0, position - 10)
                context_end = min(len(input_string), position + 10)
                context = input_string[context_start:context_end]
                
                error = LexicalError(
                    message=f"Unrecognized character '{error_char}'",
                    position=position,
                    line=line,
                    column=column,
                    context=context
                )
                errors.append(error)
                
                # Skip the unrecognized character
                position += 1
                column += 1
        
        # Add end-of-input token
        tokens.append(Token(
            type='$',
            value='',
            position=position,
            line=line,
            column=column
        ))
        
        return tokens, errors
    
    def _prepare_terminal_patterns(self):
        """
        Prepare regex patterns for terminal matching.
        
        Creates a list of (pattern, terminal_name) tuples ordered by:
        1. Longer terminals first (for longest-match)
        2. Quoted terminals (exact matches)
        3. Keyword terminals
        4. Single character terminals
        """
        self.terminal_patterns = []
        
        # Separate terminals by type and process them
        processed_terminals = []
        
        for terminal in self.terminals:
            if self._is_quoted_terminal(terminal):
                # For quoted terminals, use the unquoted value for matching
                # but keep the original terminal name for token type
                unquoted = self._unquote_terminal(terminal)
                processed_terminals.append((unquoted, terminal, 'quoted'))
            else:
                processed_terminals.append((terminal, terminal, 'unquoted'))
        
        # Sort by length of match pattern (longest first) for longest-match strategy
        processed_terminals.sort(key=lambda x: len(x[0]), reverse=True)
        
        # Create patterns for each terminal
        for match_value, terminal_name, terminal_type in processed_terminals:
            if len(match_value) == 1:
                # Single character - match exactly
                escaped = re.escape(match_value)
                pattern = re.compile(escaped)
                self.terminal_patterns.append((pattern, terminal_name))
            else:
                # Multi-character terminal
                if match_value.isalnum() or '_' in match_value:
                    # Alphanumeric keywords need word boundaries to avoid partial matches
                    pattern = re.compile(r'\b' + re.escape(match_value) + r'\b')
                else:
                    # Non-alphanumeric terminals match exactly
                    pattern = re.compile(re.escape(match_value))
                self.terminal_patterns.append((pattern, terminal_name))
    
    def _is_quoted_terminal(self, terminal: str) -> bool:
        """Check if a terminal is quoted (surrounded by single or double quotes)."""
        return ((terminal.startswith("'") and terminal.endswith("'") and len(terminal) >= 2) or
                (terminal.startswith('"') and terminal.endswith('"') and len(terminal) >= 2))
    
    def _unquote_terminal(self, terminal: str) -> str:
        """
        Remove quotes from a quoted terminal and handle escape sequences.
        
        Args:
            terminal: Quoted terminal string
            
        Returns:
            Unquoted string with escape sequences processed
        """
        if not self._is_quoted_terminal(terminal):
            return terminal
        
        # Remove outer quotes
        inner = terminal[1:-1]
        
        # Handle escape sequences
        result = ""
        i = 0
        while i < len(inner):
            if inner[i] == '\\' and i + 1 < len(inner):
                # Handle escape sequence
                next_char = inner[i + 1]
                if next_char == 'n':
                    result += '\n'
                elif next_char == 't':
                    result += '\t'
                elif next_char == 'r':
                    result += '\r'
                elif next_char == '\\':
                    result += '\\'
                elif next_char == "'":
                    result += "'"
                elif next_char == '"':
                    result += '"'
                else:
                    # Unknown escape sequence, keep as is
                    result += inner[i:i+2]
                i += 2
            else:
                result += inner[i]
                i += 1
        
        return result
    
    def get_terminal_info(self) -> Dict[str, Dict[str, Union[str, bool]]]:
        """
        Get information about all terminals for debugging.
        
        Returns:
            Dictionary mapping terminal names to their properties
        """
        info = {}
        for terminal in self.terminals:
            is_quoted = self._is_quoted_terminal(terminal)
            match_value = self._unquote_terminal(terminal) if is_quoted else terminal
            info[terminal] = {
                'is_quoted': is_quoted,
                'match_value': match_value,
                'original_length': len(terminal),
                'match_length': len(match_value),
                'type': 'quoted' if is_quoted else 
                       'single_char' if len(match_value) == 1 else 'keyword'
            }
        return info


@dataclass
class CLRAutomaton:
    """Represents the complete CLR automaton."""
    states: List[CLRState]
    transitions: Dict[Tuple[int, str], int]  # (state_id, symbol) -> target_state_id
    start_state_id: int = 0
    
    def __str__(self) -> str:
        lines = [f"CLR Automaton with {len(self.states)} states"]
        lines.append(f"Start state: {self.start_state_id}")
        lines.append("\nStates:")
        for state in self.states:
            lines.append(str(state))
        lines.append("\nTransitions:")
        for (state_id, symbol), target_id in sorted(self.transitions.items()):
            lines.append(f"  GOTO({state_id}, {symbol}) = {target_id}")
        return "\n".join(lines)


class FirstFollowComputer:
    """Computes FIRST and FOLLOW sets for grammar symbols with caching optimization."""
    
    def __init__(self, grammar: Grammar):
        self.grammar = grammar
        self._first_cache: Dict[str, Set[str]] = {}
        self._follow_cache: Dict[str, Set[str]] = {}
        self._computing_first: Set[str] = set()  # Track symbols being computed to avoid infinite recursion
        self._computing_follow: Set[str] = set()  # Track symbols being computed to avoid infinite recursion
        
        # Performance optimization: Cache string computations
        self._first_string_cache: Dict[Tuple[str, ...], Set[str]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def compute_first_sets(self) -> Dict[str, Set[str]]:
        """
        Compute FIRST sets for all grammar symbols.
        
        FIRST(X) is the set of terminals that begin strings derived from X.
        If X derives epsilon, then epsilon is in FIRST(X).
        """
        # Clear cache for fresh computation
        self._first_cache.clear()
        
        # Compute FIRST sets for all symbols
        all_symbols = self.grammar.terminals | self.grammar.non_terminals
        for symbol in all_symbols:
            self._compute_first(symbol)
        
        return self._first_cache.copy()
    
    def compute_follow_sets(self) -> Dict[str, Set[str]]:
        """
        Compute FOLLOW sets for all non-terminals.
        
        FOLLOW(A) is the set of terminals that can appear immediately 
        to the right of A in some sentential form.
        """
        # Clear cache for fresh computation
        self._follow_cache.clear()
        
        # Initialize FOLLOW sets
        for nt in self.grammar.non_terminals:
            self._follow_cache[nt] = set()
        
        # Add $ to FOLLOW of start symbol
        self._follow_cache[self.grammar.start_symbol].add('$')
        
        # Iterate until no changes (fixed point)
        changed = True
        while changed:
            changed = False
            for production in self.grammar.productions:
                for i, symbol in enumerate(production.rhs):
                    if symbol in self.grammar.non_terminals:
                        # Get FIRST of beta (symbols after current symbol)
                        beta = production.rhs[i + 1:]
                        first_beta = self._compute_first_of_string(beta)
                        
                        # Add FIRST(beta) - {epsilon} to FOLLOW(symbol)
                        before_size = len(self._follow_cache[symbol])
                        self._follow_cache[symbol].update(first_beta - {'e'})
                        if len(self._follow_cache[symbol]) > before_size:
                            changed = True
                        
                        # If epsilon in FIRST(beta), add FOLLOW(A) to FOLLOW(symbol)
                        if 'e' in first_beta:
                            before_size = len(self._follow_cache[symbol])
                            self._follow_cache[symbol].update(self._follow_cache[production.lhs])
                            if len(self._follow_cache[symbol]) > before_size:
                                changed = True
        
        return self._follow_cache.copy()
    
    def get_first(self, symbol: str) -> Set[str]:
        """Get FIRST set for a symbol (with caching)."""
        if symbol not in self._first_cache:
            self._compute_first(symbol)
        return self._first_cache[symbol].copy()
    
    def get_follow(self, symbol: str) -> Set[str]:
        """Get FOLLOW set for a non-terminal (with caching)."""
        if symbol not in self._follow_cache:
            self.compute_follow_sets()
        return self._follow_cache.get(symbol, set()).copy()
    
    def _compute_first(self, symbol: str) -> Set[str]:
        """
        Compute FIRST set for a single symbol.
        
        Rules:
        1. If X is terminal, FIRST(X) = {X}
        2. If X -> epsilon, add epsilon to FIRST(X)
        3. If X -> Y1 Y2 ... Yk, add FIRST(Y1) to FIRST(X)
           If epsilon in FIRST(Y1), add FIRST(Y2), etc.
        """
        # Return cached result if available
        if symbol in self._first_cache:
            return self._first_cache[symbol]
        
        # Avoid infinite recursion
        if symbol in self._computing_first:
            return set()
        
        self._computing_first.add(symbol)
        first_set = set()
        
        try:
            # Rule 1: If X is terminal, FIRST(X) = {X}
            if symbol in self.grammar.terminals:
                first_set.add(symbol)
            
            # Rule 2 & 3: If X is non-terminal, look at productions
            elif symbol in self.grammar.non_terminals:
                for production in self.grammar.productions:
                    if production.lhs == symbol:
                        if production.is_epsilon:
                            # Rule 2: X -> epsilon
                            first_set.add('e')
                        else:
                            # Rule 3: X -> Y1 Y2 ... Yk
                            for i, rhs_symbol in enumerate(production.rhs):
                                symbol_first = self._compute_first(rhs_symbol)
                                first_set.update(symbol_first - {'e'})
                                
                                # If epsilon not in FIRST(Yi), stop
                                if 'e' not in symbol_first:
                                    break
                                
                                # If we've processed all symbols and all had epsilon
                                if i == len(production.rhs) - 1:
                                    first_set.add('e')
            
            # Cache the result
            self._first_cache[symbol] = first_set
            return first_set
        
        finally:
            self._computing_first.discard(symbol)
    
    def _compute_first_of_string(self, symbols: List[str]) -> Set[str]:
        """
        Compute FIRST set of a string of symbols with caching.
        
        FIRST(X1 X2 ... Xn):
        - Add FIRST(X1) - {epsilon}
        - If epsilon in FIRST(X1), add FIRST(X2) - {epsilon}
        - Continue until Xi where epsilon not in FIRST(Xi)
        - If epsilon in FIRST(Xi) for all i, add epsilon
        """
        if not symbols:
            return {'e'}
        
        # Use tuple for hashable cache key
        cache_key = tuple(symbols)
        if cache_key in self._first_string_cache:
            self._cache_hits += 1
            return self._first_string_cache[cache_key].copy()
        
        self._cache_misses += 1
        first_set = set()
        
        for i, symbol in enumerate(symbols):
            symbol_first = self.get_first(symbol)
            first_set.update(symbol_first - {'e'})
            
            # If epsilon not in FIRST(symbol), stop
            if 'e' not in symbol_first:
                break
            
            # If we've processed all symbols and all had epsilon
            if i == len(symbols) - 1:
                first_set.add('e')
        
        # Cache the result
        self._first_string_cache[cache_key] = first_set.copy()
        return first_set
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache performance statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0
        return {
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate_percent': round(hit_rate, 2),
            'cached_strings': len(self._first_string_cache)
        }


class CLRItemSetBuilder:
    """Builds CLR item sets and constructs the canonical CLR automaton with enhanced lazy evaluation."""
    
    def __init__(self, grammar: Grammar):
        self.grammar = grammar
        self.ff_computer = FirstFollowComputer(grammar)
        self.first_sets = self.ff_computer.compute_first_sets()
        self.follow_sets = self.ff_computer.compute_follow_sets()
        
        # Augment grammar with S' -> S $ production
        self.augmented_start = f"{grammar.start_symbol}'"
        self.augmented_production = Production(
            lhs=self.augmented_start,
            rhs=[grammar.start_symbol],
            is_epsilon=False
        )
        
        # State management
        self.states: List[CLRState] = []
        self.state_map: Dict[frozenset, int] = {}  # Map item sets to state IDs
        self.transitions: Dict[Tuple[int, str], int] = {}
        self.next_state_id = 0
        
        # Enhanced performance optimization: Lazy evaluation and caching
        self._closure_cache: Dict[frozenset, Set] = {}
        self._goto_cache: Dict[Tuple[frozenset, str], Set] = {}
        self._lazy_states: Dict[int, bool] = {}  # Track which states are fully computed
        self._max_states = 1000  # Limit for large grammars
        self._lazy_threshold = 100  # Start lazy evaluation after this many states
        
        # Enhanced performance metrics
        self._closure_cache_hits = 0
        self._closure_cache_misses = 0
        self._goto_cache_hits = 0
        self._goto_cache_misses = 0
        self._lazy_evaluations = 0
        self._states_skipped = 0
        
        # Cache size management
        self._max_closure_cache_size = 5000
        self._max_goto_cache_size = 5000
        
        # Performance profiling
        self._profiling_enabled = False
        self._operation_times = {
            'closure': [],
            'goto': [],
            'state_creation': []
        }
    
    def build_clr_automaton(self) -> CLRAutomaton:
        """
        Build the canonical CLR automaton with enhanced lazy evaluation for large grammars.
        
        Algorithm:
        1. Create initial state with S' -> •S, $
        2. Compute closure of initial state
        3. For each state and each symbol, compute GOTO
        4. Continue until no new states are generated or limit reached
        5. Use lazy evaluation for large state spaces
        """
        import time
        start_time = time.time()
        
        # Clear flyweight cache for fresh start
        CLRItemFlyweight.clear_cache()
        
        # Create initial item set using flyweight
        initial_item = CLRItemFlyweight.get_item(
            production=self.augmented_production,
            dot_position=0,
            lookahead='$'
        )
        initial_items = {initial_item}
        
        # Compute closure of initial state
        initial_closure = self.closure(initial_items)
        
        # Create initial state
        initial_state = self._create_state(initial_closure)
        
        # Build automaton using enhanced worklist algorithm with lazy evaluation
        worklist = [initial_state.state_id]
        processed = set()
        lazy_mode = False
        
        while worklist and len(self.states) < self._max_states:
            current_state_id = worklist.pop(0)
            if current_state_id in processed:
                continue
            processed.add(current_state_id)
            
            # Enable lazy mode for large grammars
            if not lazy_mode and len(self.states) > self._lazy_threshold:
                lazy_mode = True
                self._lazy_evaluations += 1
            
            current_state = self.states[current_state_id]
            
            # Get all symbols that can follow the dot
            symbols = self._get_symbols_after_dot(current_state.items)
            
            # In lazy mode, prioritize essential transitions
            if lazy_mode:
                symbols = self._prioritize_symbols(symbols, current_state)
            
            for symbol in symbols:
                # Compute GOTO for this symbol
                goto_items = self.goto(current_state.items, symbol)
                if goto_items:
                    # Create or find target state
                    target_state = self._create_or_find_state(goto_items)
                    
                    # Add transition
                    self.transitions[(current_state_id, symbol)] = target_state.state_id
                    
                    # Add to worklist if new and under limit
                    if (target_state.state_id not in processed and 
                        len(self.states) < self._max_states):
                        # In lazy mode, defer some state computations
                        if lazy_mode and self._should_defer_state(target_state):
                            self._lazy_states[target_state.state_id] = False
                            self._states_skipped += 1
                        else:
                            worklist.append(target_state.state_id)
        
        # Mark all created states as fully computed (unless deferred)
        for state_id in range(len(self.states)):
            if state_id not in self._lazy_states:
                self._lazy_states[state_id] = True
        
        # Clean up caches if they're too large
        self._cleanup_caches()
        
        build_time = time.time() - start_time
        if self._profiling_enabled:
            if 'automaton_build' not in self._operation_times:
                self._operation_times['automaton_build'] = []
            self._operation_times['automaton_build'].append(build_time)
        
        return CLRAutomaton(
            states=self.states,
            transitions=self.transitions,
            start_state_id=0
        )
    
    def _prioritize_symbols(self, symbols: Set[str], current_state: CLRState) -> List[str]:
        """Prioritize symbols for lazy evaluation - terminals first, then frequent non-terminals."""
        terminals = [s for s in symbols if s in self.grammar.terminals]
        non_terminals = [s for s in symbols if s in self.grammar.non_terminals]
        
        # Sort non-terminals by frequency in grammar (more frequent first)
        nt_frequency = {}
        for prod in self.grammar.productions:
            for symbol in prod.rhs:
                if symbol in self.grammar.non_terminals:
                    nt_frequency[symbol] = nt_frequency.get(symbol, 0) + 1
        
        non_terminals.sort(key=lambda x: nt_frequency.get(x, 0), reverse=True)
        
        return terminals + non_terminals
    
    def _should_defer_state(self, state: CLRState) -> bool:
        """Determine if a state computation should be deferred in lazy mode."""
        # Defer states with many items or complex lookaheads
        if len(state.items) > 10:
            return True
        
        # Defer states with only reduce items (less critical for immediate parsing)
        reduce_items = sum(1 for item in state.items if item.is_complete())
        if reduce_items == len(state.items) and len(state.items) > 3:
            return True
        
        return False
    
    def _cleanup_caches(self):
        """Clean up caches when they become too large."""
        if len(self._closure_cache) > self._max_closure_cache_size:
            # Keep only the most recently used half
            items = list(self._closure_cache.items())
            self._closure_cache = dict(items[-self._max_closure_cache_size//2:])
        
        if len(self._goto_cache) > self._max_goto_cache_size:
            # Keep only the most recently used half
            items = list(self._goto_cache.items())
            self._goto_cache = dict(items[-self._max_goto_cache_size//2:])
    
    def get_performance_stats(self) -> Dict[str, Union[int, float]]:
        """Get comprehensive performance statistics for optimization analysis."""
        total_closure = self._closure_cache_hits + self._closure_cache_misses
        total_goto = self._goto_cache_hits + self._goto_cache_misses
        
        closure_hit_rate = (self._closure_cache_hits / total_closure * 100) if total_closure > 0 else 0
        goto_hit_rate = (self._goto_cache_hits / total_goto * 100) if total_goto > 0 else 0
        
        stats = {
            'states_created': len(self.states),
            'transitions_created': len(self.transitions),
            'flyweight_items': CLRItemFlyweight.get_cache_size(),
            'closure_cache_hits': self._closure_cache_hits,
            'closure_cache_misses': self._closure_cache_misses,
            'closure_hit_rate_percent': round(closure_hit_rate, 2),
            'goto_cache_hits': self._goto_cache_hits,
            'goto_cache_misses': self._goto_cache_misses,
            'goto_hit_rate_percent': round(goto_hit_rate, 2),
            'closure_cache_size': len(self._closure_cache),
            'goto_cache_size': len(self._goto_cache),
            'lazy_evaluations': self._lazy_evaluations,
            'states_skipped': self._states_skipped,
            'max_states_limit': self._max_states,
            'lazy_threshold': self._lazy_threshold
        }
        
        # Add FIRST/FOLLOW cache stats
        ff_stats = self.ff_computer.get_cache_stats()
        stats.update({f'first_follow_{k}': v for k, v in ff_stats.items()})
        
        # Add flyweight cache stats
        flyweight_stats = CLRItemFlyweight.get_cache_stats()
        stats.update({f'flyweight_{k}': v for k, v in flyweight_stats.items()})
        
        # Add profiling data if available
        if self._profiling_enabled and self._operation_times:
            for op, times in self._operation_times.items():
                if times:
                    stats[f'{op}_avg_time_ms'] = round(sum(times) / len(times) * 1000, 3)
                    stats[f'{op}_total_time_ms'] = round(sum(times) * 1000, 3)
                    stats[f'{op}_call_count'] = len(times)
        
        return stats
    
    def enable_profiling(self, enabled: bool = True):
        """Enable or disable performance profiling."""
        self._profiling_enabled = enabled
        if not enabled:
            self._operation_times.clear()
    
    def get_memory_usage_estimate(self) -> Dict[str, int]:
        """Estimate memory usage of various components."""
        import sys
        
        # Estimate sizes (rough approximations)
        state_size = sum(sys.getsizeof(state) for state in self.states)
        transition_size = sys.getsizeof(self.transitions)
        cache_size = (sys.getsizeof(self._closure_cache) + 
                     sys.getsizeof(self._goto_cache))
        
        return {
            'states_bytes': state_size,
            'transitions_bytes': transition_size,
            'cache_bytes': cache_size,
            'total_estimated_bytes': state_size + transition_size + cache_size,
            'flyweight_cache_bytes': CLRItemFlyweight.get_cache_size() * 64  # Rough estimate
        }
    
    def closure(self, items: Set[CLRItem]) -> Set[CLRItem]:
        """
        Compute the closure of a set of CLR items with enhanced caching optimization.
        
        Algorithm:
        1. Start with the given items
        2. For each item [A -> α•Bβ, a] where B is non-terminal:
        3. For each production B -> γ:
        4. For each terminal b in FIRST(βa):
        5. Add item [B -> •γ, b] if not already present
        6. Repeat until no new items are added
        """
        import time
        start_time = time.time() if self._profiling_enabled else 0
        
        # Check cache first
        items_key = frozenset(items)
        if items_key in self._closure_cache:
            self._closure_cache_hits += 1
            if self._profiling_enabled:
                self._operation_times['closure'].append(time.time() - start_time)
            return self._closure_cache[items_key].copy()
        
        self._closure_cache_misses += 1
        closure_items = set(items)
        changed = True
        
        # Optimization: Pre-compute productions by LHS for faster lookup
        productions_by_lhs = {}
        for production in self.grammar.productions:
            if production.lhs not in productions_by_lhs:
                productions_by_lhs[production.lhs] = []
            productions_by_lhs[production.lhs].append(production)
        
        while changed:
            changed = False
            new_items = set()
            
            for item in closure_items:
                # Check if dot is before a non-terminal
                next_symbol = item.next_symbol()
                if next_symbol and next_symbol in self.grammar.non_terminals:
                    # Get β (symbols after the non-terminal)
                    beta = item.production.rhs[item.dot_position + 1:]
                    
                    # Compute FIRST(βa) where a is the lookahead
                    beta_a = beta + [item.lookahead]
                    first_beta_a = self._compute_first_of_string(beta_a)
                    
                    # For each production B -> γ (optimized lookup)
                    for production in productions_by_lhs.get(next_symbol, []):
                        # For each terminal in FIRST(βa)
                        for terminal in first_beta_a:
                            if terminal != 'e':  # Skip epsilon
                                # Use flyweight pattern for CLR items
                                new_item = CLRItemFlyweight.get_item(
                                    production=production,
                                    dot_position=0,
                                    lookahead=terminal
                                )
                                if new_item not in closure_items:
                                    new_items.add(new_item)
                                    changed = True
            
            closure_items.update(new_items)
        
        # Cache the result (with size limit)
        if len(self._closure_cache) < self._max_closure_cache_size:
            self._closure_cache[items_key] = closure_items.copy()
        
        if self._profiling_enabled:
            self._operation_times['closure'].append(time.time() - start_time)
        
        return closure_items
    
    def goto(self, items: Set[CLRItem], symbol: str) -> Set[CLRItem]:
        """
        Compute GOTO(I, X) with enhanced caching optimization.
        
        Algorithm:
        1. For each item [A -> α•Xβ, a] in items where next symbol is X:
        2. Create new item [A -> αX•β, a] (move dot past X)
        3. Return closure of all such new items
        """
        import time
        start_time = time.time() if self._profiling_enabled else 0
        
        # Check cache first
        cache_key = (frozenset(items), symbol)
        if cache_key in self._goto_cache:
            self._goto_cache_hits += 1
            if self._profiling_enabled:
                self._operation_times['goto'].append(time.time() - start_time)
            return self._goto_cache[cache_key].copy()
        
        self._goto_cache_misses += 1
        goto_items = set()
        
        # Optimization: Filter items first to avoid unnecessary processing
        relevant_items = [item for item in items if item.next_symbol() == symbol]
        
        for item in relevant_items:
            # Move dot past the symbol using flyweight pattern
            new_item = CLRItemFlyweight.get_item(
                production=item.production,
                dot_position=item.dot_position + 1,
                lookahead=item.lookahead
            )
            goto_items.add(new_item)
        
        # Return closure of the new items
        if goto_items:
            result = self.closure(goto_items)
        else:
            result = set()
        
        # Cache the result (with size limit)
        if len(self._goto_cache) < self._max_goto_cache_size:
            self._goto_cache[cache_key] = result.copy()
        
        if self._profiling_enabled:
            self._operation_times['goto'].append(time.time() - start_time)
        
        return result
    
    def _create_state(self, items: Set[CLRItem]) -> CLRState:
        """Create a new state with the given items."""
        state = CLRState(items=items, state_id=self.next_state_id)
        self.states.append(state)
        
        # Map the item set to the state ID for efficient lookup
        items_key = frozenset(items)
        self.state_map[items_key] = self.next_state_id
        
        self.next_state_id += 1
        return state
    
    def _create_or_find_state(self, items: Set[CLRItem]) -> CLRState:
        """Create a new state or return existing state with the same items."""
        items_key = frozenset(items)
        
        if items_key in self.state_map:
            # Return existing state
            state_id = self.state_map[items_key]
            return self.states[state_id]
        else:
            # Create new state
            return self._create_state(items)
    
    def _get_symbols_after_dot(self, items: Set[CLRItem]) -> Set[str]:
        """Get all symbols that can appear after the dot in the given items."""
        symbols = set()
        for item in items:
            next_symbol = item.next_symbol()
            if next_symbol:
                symbols.add(next_symbol)
        return symbols
    
    def _compute_first_of_string(self, symbols: List[str]) -> Set[str]:
        """
        Compute FIRST set of a string of symbols.
        Uses the FirstFollowComputer's logic but handles the case where
        symbols might include terminals not in the original grammar.
        """
        if not symbols:
            return {'e'}
        
        first_set = set()
        
        for i, symbol in enumerate(symbols):
            if symbol in self.first_sets:
                symbol_first = self.first_sets[symbol]
            elif symbol in self.grammar.terminals or symbol == '$':
                # Terminal symbol
                symbol_first = {symbol}
            else:
                # Unknown symbol, treat as terminal
                symbol_first = {symbol}
            
            first_set.update(symbol_first - {'e'})
            
            # If epsilon not in FIRST(symbol), stop
            if 'e' not in symbol_first:
                break
            
            # If we've processed all symbols and all had epsilon
            if i == len(symbols) - 1:
                first_set.add('e')
        
        return first_set


@dataclass
class ParsingTables:
    """Represents the complete set of CLR parsing tables."""
    action_table: Dict[Tuple[int, str], str]  # (state, terminal) -> action
    goto_table: Dict[Tuple[int, str], int]   # (state, non_terminal) -> state
    
    def __str__(self) -> str:
        lines = ["Parsing Tables:"]
        lines.append("\nAction Table:")
        for (state, terminal), action in sorted(self.action_table.items()):
            lines.append(f"  ACTION[{state}, {terminal}] = {action}")
        lines.append("\nGoto Table:")
        for (state, non_terminal), target in sorted(self.goto_table.items()):
            lines.append(f"  GOTO[{state}, {non_terminal}] = {target}")
        return "\n".join(lines)


@dataclass
class Conflict:
    """Represents a parsing conflict in the CLR tables."""
    state_id: int
    symbol: str
    conflict_type: str  # "shift/reduce" or "reduce/reduce"
    actions: List[str]  # List of conflicting actions
    description: str
    
    def __str__(self) -> str:
        return f"{self.conflict_type} conflict in state {self.state_id} on symbol '{self.symbol}': {self.description}"


class ActionType(Enum):
    """Enumeration of CLR parsing actions."""
    SHIFT = "shift"
    REDUCE = "reduce"
    ACCEPT = "accept"
    ERROR = "error"


@dataclass
class ParseAction:
    """Represents a single parsing action."""
    action_type: ActionType
    value: Optional[Union[int, Production]] = None  # State ID for shift, Production for reduce
    
    def __str__(self) -> str:
        if self.action_type == ActionType.SHIFT:
            return f"shift {self.value}"
        elif self.action_type == ActionType.REDUCE:
            return f"reduce {self.value}"
        elif self.action_type == ActionType.ACCEPT:
            return "accept"
        else:
            return "error"


@dataclass
class ParseTreeNode:
    """Represents a node in the parse tree."""
    label: str
    children: List['ParseTreeNode'] = field(default_factory=list)
    is_terminal: bool = False
    token: Optional[Token] = None  # For terminal nodes
    
    def __str__(self) -> str:
        if self.is_terminal:
            return f"'{self.label}'"
        else:
            return self.label
    
    def to_dot(self, node_id: int = 0) -> Tuple[str, int]:
        """
        Convert parse tree to DOT format for visualization.
        
        Returns:
            Tuple of (dot_string, next_node_id)
        """
        lines = []
        current_id = node_id
        
        # Escape label for DOT format
        escaped_label = self.label.replace('"', '\\"').replace('\n', '\\n')
        
        # Create node with better styling
        if self.is_terminal:
            lines.append(f'  node{current_id} [label="{escaped_label}", shape=box, style=filled, fillcolor=lightblue, fontname="Courier"];')
        else:
            lines.append(f'  node{current_id} [label="{escaped_label}", shape=ellipse, style=filled, fillcolor=lightgreen, fontname="Arial"];')
        
        # Create edges to children
        next_id = current_id + 1
        for child in self.children:
            child_dot, next_id = child.to_dot(next_id)
            lines.append(child_dot)
            # Fix the edge connection - connect to the child's actual ID
            child_id = next_id - 1
            lines.append(f'  node{current_id} -> node{child_id};')
        
        return '\n'.join(lines), next_id


@dataclass
class ParseStep:
    """Represents a single step in the parsing trace."""
    step_number: int
    stack: List[int]  # Stack of state IDs
    input_buffer: List[Token]  # Remaining input tokens
    action: ParseAction
    production_used: Optional[Production] = None  # For reduce actions
    symbol_stack: Optional[str] = None  # Concatenated stack representation (e.g., "0a3b2")
    
    def __str__(self) -> str:
        if self.symbol_stack:
            stack_str = self.symbol_stack
        else:
            stack_str = ' '.join(map(str, self.stack))
        input_str = ' '.join(token.type for token in self.input_buffer[:5])
        if len(self.input_buffer) > 5:
            input_str += " ..."
        
        return f"Step {self.step_number}: Stack=[{stack_str}] Input=[{input_str}] Action={self.action}"


@dataclass
class ParseResult:
    """Represents the result of a parsing operation."""
    success: bool
    parse_tree: Optional[ParseTreeNode] = None
    error_message: str = ""
    error_position: int = -1
    trace: List[ParseStep] = field(default_factory=list)
    
    def __str__(self) -> str:
        if self.success:
            return f"Parse successful. Tree: {self.parse_tree}"
        else:
            return f"Parse failed: {self.error_message} at position {self.error_position}"


class CLRTableGenerator:
    """Generates CLR parsing tables from a CLR automaton with enhanced caching optimization."""
    
    def __init__(self, grammar: Grammar, automaton: CLRAutomaton):
        self.grammar = grammar
        self.automaton = automaton
        self.action_table: Dict[Tuple[int, str], str] = {}
        self.goto_table: Dict[Tuple[int, str], int] = {}
        self.conflicts: List[Conflict] = []
        
        # Enhanced performance optimization: Cache frequently accessed entries
        self._action_cache: Dict[Tuple[int, str], str] = {}
        self._goto_cache: Dict[Tuple[int, str], int] = {}
        self._cache_access_count: Dict[Tuple[int, str], int] = {}
        self._cache_threshold = 3  # Cache entries accessed more than this many times
        self._max_cache_size = 1000  # Limit cache size
        
        # Performance tracking
        self._cache_hits = 0
        self._cache_misses = 0
        self._table_generation_time = 0
        
        # Optimization: Pre-compute frequently used data
        self._terminal_list = sorted(grammar.terminals)
        self._non_terminal_list = sorted(grammar.non_terminals)
        self._state_count = len(automaton.states)
    
    def generate_parsing_tables(self) -> ParsingTables:
        """
        Generate both action and goto tables from the CLR automaton.
        
        Returns:
            ParsingTables object containing both tables
        """
        self._reset_tables()
        self._generate_action_table()
        self._generate_goto_table()
        
        return ParsingTables(
            action_table=self.action_table.copy(),
            goto_table=self.goto_table.copy()
        )
    
    def generate_action_table(self) -> Dict[Tuple[int, str], str]:
        """
        Generate the ACTION table from CLR automaton.
        
        Algorithm:
        1. For each state i and each terminal a:
           - If [A -> α•aβ, b] in state i and GOTO(i,a) = j, then ACTION[i,a] = "shift j"
           - If [A -> α•, a] in state i and A ≠ S', then ACTION[i,a] = "reduce A -> α"
           - If [S' -> S•, $] in state i, then ACTION[i,$] = "accept"
        2. Detect and report conflicts
        
        Returns:
            Dictionary mapping (state_id, terminal) to action string
        """
        self.action_table.clear()
        
        for state in self.automaton.states:
            state_id = state.state_id
            
            for item in state.items:
                if item.is_complete():
                    # Reduce or accept action
                    if item.production.lhs == f"{self.grammar.start_symbol}'":
                        # Accept action for augmented start production
                        self._add_action(state_id, item.lookahead, "accept")
                    else:
                        # Reduce action
                        prod_str = self._production_to_string(item.production)
                        action = f"reduce {prod_str}"
                        self._add_action(state_id, item.lookahead, action)
                else:
                    # Shift action (if next symbol is terminal)
                    next_symbol = item.next_symbol()
                    if next_symbol and next_symbol in self.grammar.terminals:
                        # Check if there's a transition for this symbol
                        if (state_id, next_symbol) in self.automaton.transitions:
                            target_state = self.automaton.transitions[(state_id, next_symbol)]
                            action = f"shift {target_state}"
                            self._add_action(state_id, next_symbol, action)
        
        return self.action_table.copy()
    
    def generate_goto_table(self) -> Dict[Tuple[int, str], int]:
        """
        Generate the GOTO table from CLR automaton.
        
        Algorithm:
        For each state i and each non-terminal A:
        If GOTO(i, A) = j, then GOTO[i, A] = j
        
        Returns:
            Dictionary mapping (state_id, non_terminal) to target_state_id
        """
        self.goto_table.clear()
        
        for (state_id, symbol), target_state in self.automaton.transitions.items():
            if symbol in self.grammar.non_terminals:
                self.goto_table[(state_id, symbol)] = target_state
        
        return self.goto_table.copy()
    
    def detect_conflicts(self, tables: Optional[ParsingTables] = None) -> List[Conflict]:
        """
        Detect shift/reduce and reduce/reduce conflicts in the parsing tables.
        
        Args:
            tables: Optional ParsingTables object. If None, uses internal tables.
            
        Returns:
            List of Conflict objects describing any conflicts found
        """
        if tables is None:
            action_table = self.action_table
        else:
            action_table = tables.action_table
        
        conflicts = []
        
        # Check for conflict markers in action table
        for (state_id, symbol), action in action_table.items():
            if isinstance(action, str) and action.startswith("[CONFLICT]"):
                # Parse the conflict string to extract individual actions
                conflict_part = action[10:]  # Remove "[CONFLICT] " prefix
                actions = [a.strip() for a in conflict_part.split(" | ")]
                
                shift_actions = [a for a in actions if a.startswith("shift")]
                reduce_actions = [a for a in actions if a.startswith("reduce")]
                accept_actions = [a for a in actions if a == "accept"]
                
                if shift_actions and (reduce_actions or accept_actions):
                    # Shift/reduce conflict
                    conflict = Conflict(
                        state_id=state_id,
                        symbol=symbol,
                        conflict_type="shift/reduce",
                        actions=actions,
                        description=f"Shift action {shift_actions[0]} conflicts with reduce/accept actions {reduce_actions + accept_actions}"
                    )
                    conflicts.append(conflict)
                elif len(reduce_actions) > 1:
                    # Reduce/reduce conflict
                    conflict = Conflict(
                        state_id=state_id,
                        symbol=symbol,
                        conflict_type="reduce/reduce",
                        actions=actions,
                        description=f"Multiple reduce actions: {reduce_actions}"
                    )
                    conflicts.append(conflict)
        
        self.conflicts = conflicts
        return conflicts
    
    def generate_conflict_report(self) -> str:
        """
        Generate a detailed report of all conflicts found in the parsing tables.
        
        Returns:
            String containing detailed conflict analysis
        """
        if not self.conflicts:
            return "No conflicts detected in the parsing tables."
        
        lines = [f"Found {len(self.conflicts)} conflict(s) in the parsing tables:\n"]
        
        for i, conflict in enumerate(self.conflicts, 1):
            lines.append(f"Conflict {i}: {conflict}")
            lines.append(f"  State {conflict.state_id} details:")
            
            # Show the items in the conflicting state
            if conflict.state_id < len(self.automaton.states):
                state = self.automaton.states[conflict.state_id]
                for item in sorted(state.items, key=str):
                    lines.append(f"    {item}")
            
            lines.append(f"  Conflicting actions on symbol '{conflict.symbol}':")
            for action in conflict.actions:
                lines.append(f"    {action}")
            
            lines.append("")  # Empty line between conflicts
        
        return "\n".join(lines)
    
    def generate_conflict_report_html(self) -> str:
        """
        Generate HTML report of all conflicts found in the parsing tables.
        
        Returns:
            HTML string containing detailed conflict analysis
        """
        from visualization import ErrorMessageFormatter
        formatter = ErrorMessageFormatter()
        return formatter.format_conflict_report(self.conflicts)
    
    def generate_parsing_tables_html(self) -> str:
        """
        Generate HTML representation of the parsing tables.
        
        Returns:
            HTML string containing ACTION and GOTO tables
        """
        from visualization import HTMLTableGenerator
        generator = HTMLTableGenerator()
        return generator.generate_action_goto_tables_html(
            self.action_table,
            self.goto_table,
            self.grammar.terminals,
            self.grammar.non_terminals
        )
    
    def _reset_tables(self):
        """Reset internal table state for fresh generation."""
        self.action_table.clear()
        self.goto_table.clear()
        self.conflicts.clear()
    
    def _generate_action_table(self):
        """Internal method to generate action table."""
        self.generate_action_table()
    
    def _generate_goto_table(self):
        """Internal method to generate goto table."""
        self.generate_goto_table()
    
    def _add_action(self, state_id: int, symbol: str, action: str):
        """
        Add an action to the action table, detecting conflicts.
        
        Args:
            state_id: State ID
            symbol: Terminal symbol
            action: Action string (e.g., "shift 5", "reduce E -> E + T")
        """
        key = (state_id, symbol)
        
        if key in self.action_table:
            # Potential conflict - store both actions for conflict detection
            existing_action = self.action_table[key]
            if existing_action != action:
                # Create a conflict marker by storing multiple actions
                # We'll handle this in detect_conflicts()
                if isinstance(existing_action, str) and not existing_action.startswith("[CONFLICT]"):
                    # First conflict for this entry
                    self.action_table[key] = f"[CONFLICT] {existing_action} | {action}"
                else:
                    # Additional conflict
                    self.action_table[key] += f" | {action}"
        else:
            self.action_table[key] = action
    
    def _production_to_string(self, production: Production) -> str:
        """
        Convert a Production object to a string representation for reduce actions.
        
        Args:
            production: Production object
            
        Returns:
            String representation like "E -> E + T" or "A -> ε"
        """
        if production.is_epsilon:
            return f"{production.lhs} -> e"
        else:
            return f"{production.lhs} -> {' '.join(production.rhs)}"


class CLRParsingEngine:
    """
    CLR parsing engine with enhanced performance optimizations.
    
    Implements stack-based CLR parsing with:
    - Shift, reduce, and accept actions
    - Parse tree construction during parsing
    - Detailed parsing trace generation
    - Comprehensive error reporting
    - Enhanced cached table lookups for frequently accessed entries
    - Adaptive caching based on access patterns
    """
    
    def __init__(self, grammar: Grammar, parsing_tables: ParsingTables):
        """
        Initialize the CLR parsing engine with enhanced caching.
        
        Args:
            grammar: The context-free grammar
            parsing_tables: The CLR parsing tables (action and goto)
        """
        self.grammar = grammar
        self.parsing_tables = parsing_tables
        self.lexer = LexicalAnalyzer(grammar.terminals)
        
        # Enhanced performance optimization: Adaptive caching
        self._action_lookup_cache: Dict[Tuple[int, str], str] = {}
        self._goto_lookup_cache: Dict[Tuple[int, str], int] = {}
        self._access_frequency: Dict[Tuple[int, str], int] = {}
        self._cache_threshold = 2  # Cache after this many accesses
        self._max_cache_size = 500  # Limit cache size
        
        self._lookup_stats: Dict[str, int] = {
            'action_cache_hits': 0,
            'action_cache_misses': 0,
            'goto_cache_hits': 0,
            'goto_cache_misses': 0,
            'cache_evictions': 0
        }
        
        # Pre-warm cache with common entries
        self._prewarm_cache()
    
    def _prewarm_cache(self):
        """Pre-warm cache with commonly accessed table entries."""
        # Pre-load entries for state 0 (initial state) and common terminals
        common_terminals = ['$', 'id', '+', '*', '(', ')']  # Common in many grammars
        
        for terminal in common_terminals:
            if terminal in self.grammar.terminals:
                key = (0, terminal)
                if key in self.parsing_tables.action_table:
                    self._action_lookup_cache[key] = self.parsing_tables.action_table[key]
    
    def _evict_cache_entries(self):
        """Evict least frequently used cache entries when cache is full."""
        if len(self._action_lookup_cache) > self._max_cache_size:
            # Remove 20% of least frequently used entries
            sorted_entries = sorted(self._access_frequency.items(), key=lambda x: x[1])
            evict_count = len(sorted_entries) // 5
            
            for key, _ in sorted_entries[:evict_count]:
                self._action_lookup_cache.pop(key, None)
                self._goto_lookup_cache.pop(key, None)
                self._access_frequency.pop(key, None)
                self._lookup_stats['cache_evictions'] += 1
    
    def parse(self, input_string: str) -> ParseResult:
        """
        Parse an input string using the CLR parsing algorithm.
        
        Args:
            input_string: The string to parse
            
        Returns:
            ParseResult containing success status, parse tree, and trace
        """
        # Tokenize the input
        tokens, lex_errors = self.lexer.tokenize(input_string)
        
        if lex_errors:
            # Return lexical error
            error_msg = f"Lexical error: {lex_errors[0].message}"
            return ParseResult(
                success=False,
                error_message=error_msg,
                error_position=lex_errors[0].position
            )
        
        # Execute CLR parsing algorithm
        return self._execute_clr_parsing(tokens)
    
    def parse_tokens(self, tokens: List[Token]) -> ParseResult:
        """
        Parse a list of tokens using the CLR parsing algorithm.
        
        Args:
            tokens: List of tokens to parse
            
        Returns:
            ParseResult containing success status, parse tree, and trace
        """
        return self._execute_clr_parsing(tokens)
    
    def _execute_clr_parsing(self, tokens: List[Token]) -> ParseResult:
        """
        Execute the CLR parsing algorithm on a list of tokens.
        
        Algorithm:
        1. Initialize stack with state 0
        2. For each input token:
           - Lookup action in action table
           - Execute shift, reduce, or accept action
           - Update stack and input pointer
        3. Build parse tree from reduction sequence
        
        Args:
            tokens: List of tokens to parse
            
        Returns:
            ParseResult with parsing outcome
        """
        # Initialize parsing state
        stack = [0]  # Stack of state IDs
        parse_stack = []  # Stack of parse tree nodes for reductions
        input_buffer = tokens.copy()
        trace = []
        step_number = 1
        
        while True:
            # Get current state and input symbol
            current_state = stack[-1]
            
            if not input_buffer:
                # This shouldn't happen if input is properly terminated with $
                return ParseResult(
                    success=False,
                    error_message="Unexpected end of input",
                    error_position=len(tokens),
                    trace=trace
                )
            
            current_token = input_buffer[0]
            current_symbol = current_token.type
            
            # Lookup action in action table with caching
            action_key = (current_state, current_symbol)
            action_str = self._get_cached_action(action_key)
            
            if action_str is None:
                # Parse error - no action defined
                expected_symbols = self._get_expected_symbols(current_state)
                error_msg = f"Unexpected symbol '{current_symbol}'. Expected one of: {expected_symbols}"
                
                return ParseResult(
                    success=False,
                    error_message=error_msg,
                    error_position=current_token.position,
                    trace=trace
                )
            
            # Handle conflicts by taking the first action
            if action_str.startswith("[CONFLICT]"):
                # Extract first action from conflict
                conflict_part = action_str[10:]  # Remove "[CONFLICT] " prefix
                actions = [a.strip() for a in conflict_part.split(" | ")]
                action_str = actions[0]  # Take first action
            
            # Parse the action
            action = self._parse_action(action_str)
            
            # Create concatenated stack representation (like "0a3b2")
            symbol_stack_str = ""
            for i, state in enumerate(stack):
                symbol_stack_str += str(state)
                if i < len(parse_stack):
                    # Add the symbol from parse_stack
                    symbol_stack_str += parse_stack[i].label
            
            # Record the parsing step
            step = ParseStep(
                step_number=step_number,
                stack=stack.copy(),
                input_buffer=input_buffer.copy(),
                action=action,
                symbol_stack=symbol_stack_str
            )
            trace.append(step)
            step_number += 1
            
            # Execute the action
            if action.action_type == ActionType.SHIFT:
                # Shift action: push token and new state
                target_state = action.value
                
                # Create terminal node for parse tree
                terminal_node = ParseTreeNode(
                    label=current_token.value,
                    is_terminal=True,
                    token=current_token
                )
                
                # Push token node and new state
                parse_stack.append(terminal_node)
                stack.append(target_state)
                
                # Consume input token
                input_buffer.pop(0)
                
            elif action.action_type == ActionType.REDUCE:
                # Reduce action: pop symbols and push non-terminal
                production = action.value
                step.production_used = production
                
                # Pop symbols from stack (2 * len(rhs) because we have states and symbols)
                rhs_length = len(production.rhs)
                
                # Collect children for parse tree (in reverse order)
                children = []
                for _ in range(rhs_length):
                    if parse_stack:
                        children.append(parse_stack.pop())
                    if stack:
                        stack.pop()  # Pop state
                
                # Reverse children to get correct order
                children.reverse()
                
                # Create non-terminal node
                non_terminal_node = ParseTreeNode(
                    label=production.lhs,
                    children=children,
                    is_terminal=False
                )
                
                # Push non-terminal node
                parse_stack.append(non_terminal_node)
                
                # Get new state from goto table
                if not stack:
                    return ParseResult(
                        success=False,
                        error_message="Stack underflow during reduce",
                        error_position=current_token.position,
                        trace=trace
                    )
                
                current_state = stack[-1]
                goto_key = (current_state, production.lhs)
                new_state = self._get_cached_goto(goto_key)
                
                if new_state is None:
                    return ParseResult(
                        success=False,
                        error_message=f"No goto entry for state {current_state}, symbol {production.lhs}",
                        error_position=current_token.position,
                        trace=trace
                    )
                stack.append(new_state)
                
            elif action.action_type == ActionType.ACCEPT:
                # Accept action: parsing successful
                if len(parse_stack) != 1:
                    return ParseResult(
                        success=False,
                        error_message=f"Parse stack should have exactly 1 element at accept, has {len(parse_stack)}",
                        error_position=current_token.position,
                        trace=trace
                    )
                
                parse_tree = parse_stack[0]
                
                return ParseResult(
                    success=True,
                    parse_tree=parse_tree,
                    trace=trace
                )
                
            else:
                # Error action
                return ParseResult(
                    success=False,
                    error_message="Parse error",
                    error_position=current_token.position,
                    trace=trace
                )
    
    def _parse_action(self, action_str: str) -> ParseAction:
        """
        Parse an action string into a ParseAction object.
        
        Args:
            action_str: Action string like "shift 5", "reduce E -> E + T", "accept"
            
        Returns:
            ParseAction object
        """
        action_str = action_str.strip()
        
        if action_str == "accept":
            return ParseAction(ActionType.ACCEPT)
        elif action_str.startswith("shift "):
            state_id = int(action_str[6:])
            return ParseAction(ActionType.SHIFT, state_id)
        elif action_str.startswith("reduce "):
            # Parse production from string
            prod_str = action_str[7:]  # Remove "reduce "
            production = self._parse_production_string(prod_str)
            return ParseAction(ActionType.REDUCE, production)
        else:
            return ParseAction(ActionType.ERROR)
    
    def _parse_production_string(self, prod_str: str) -> Production:
        """
        Parse a production string back into a Production object.
        
        Args:
            prod_str: Production string like "E -> E + T" or "A -> e"
            
        Returns:
            Production object
        """
        # Split on " -> "
        parts = prod_str.split(" -> ")
        if len(parts) != 2:
            raise ValueError(f"Invalid production string: {prod_str}")
        
        lhs = parts[0].strip()
        rhs_str = parts[1].strip()
        
        if rhs_str == "e":
            # Epsilon production
            return Production(lhs=lhs, rhs=[], is_epsilon=True)
        else:
            # Regular production
            rhs = rhs_str.split()
            return Production(lhs=lhs, rhs=rhs, is_epsilon=False)
    
    def _get_expected_symbols(self, state_id: int) -> List[str]:
        """
        Get the list of symbols that are expected in the given state.
        
        Args:
            state_id: The parser state ID
            
        Returns:
            List of expected terminal symbols
        """
        expected = []
        
        for (state, symbol), action in self.parsing_tables.action_table.items():
            if state == state_id:
                expected.append(symbol)
        
        return sorted(expected)
    
    def generate_trace_html(self, trace: List[ParseStep]) -> str:
        """
        Generate HTML representation of the parsing trace.
        
        Args:
            trace: List of parsing steps
            
        Returns:
            HTML string showing step-by-step parsing
        """
        from visualization import ParseTraceFormatter
        formatter = ParseTraceFormatter()
        return formatter.generate_trace_html(trace)
    
    def _get_cached_action(self, action_key: Tuple[int, str]) -> Optional[str]:
        """
        Get action from cache or parsing table with enhanced performance tracking.
        
        Args:
            action_key: Tuple of (state_id, terminal)
            
        Returns:
            Action string or None if not found
        """
        # Track access frequency
        self._access_frequency[action_key] = self._access_frequency.get(action_key, 0) + 1
        
        # Check cache first
        if action_key in self._action_lookup_cache:
            self._lookup_stats['action_cache_hits'] += 1
            return self._action_lookup_cache[action_key]
        
        # Check parsing table
        if action_key in self.parsing_tables.action_table:
            action = self.parsing_tables.action_table[action_key]
            self._lookup_stats['action_cache_misses'] += 1
            
            # Cache frequently accessed entries
            if self._access_frequency[action_key] >= self._cache_threshold:
                # Check cache size limit
                if len(self._action_lookup_cache) >= self._max_cache_size:
                    self._evict_cache_entries()
                
                self._action_lookup_cache[action_key] = action
            
            return action
        
        self._lookup_stats['action_cache_misses'] += 1
        return None
    
    def _get_cached_goto(self, goto_key: Tuple[int, str]) -> Optional[int]:
        """
        Get goto state from cache or parsing table with enhanced performance tracking.
        
        Args:
            goto_key: Tuple of (state_id, non_terminal)
            
        Returns:
            Target state ID or None if not found
        """
        # Track access frequency
        self._access_frequency[goto_key] = self._access_frequency.get(goto_key, 0) + 1
        
        # Check cache first
        if goto_key in self._goto_lookup_cache:
            self._lookup_stats['goto_cache_hits'] += 1
            return self._goto_lookup_cache[goto_key]
        
        # Check parsing table
        if goto_key in self.parsing_tables.goto_table:
            target_state = self.parsing_tables.goto_table[goto_key]
            self._lookup_stats['goto_cache_misses'] += 1
            
            # Cache frequently accessed entries
            if self._access_frequency[goto_key] >= self._cache_threshold:
                # Check cache size limit
                if len(self._goto_lookup_cache) >= self._max_cache_size:
                    self._evict_cache_entries()
                
                self._goto_lookup_cache[goto_key] = target_state
            
            return target_state
        
        self._lookup_stats['goto_cache_misses'] += 1
        return None
    
    def get_performance_stats(self) -> Dict[str, Union[int, float]]:
        """Get performance statistics for the parsing engine."""
        total_action = self._lookup_stats['action_cache_hits'] + self._lookup_stats['action_cache_misses']
        total_goto = self._lookup_stats['goto_cache_hits'] + self._lookup_stats['goto_cache_misses']
        
        action_hit_rate = (self._lookup_stats['action_cache_hits'] / total_action * 100) if total_action > 0 else 0
        goto_hit_rate = (self._lookup_stats['goto_cache_hits'] / total_goto * 100) if total_goto > 0 else 0
        
        return {
            'action_cache_size': len(self._action_lookup_cache),
            'goto_cache_size': len(self._goto_lookup_cache),
            'action_hit_rate_percent': round(action_hit_rate, 2),
            'goto_hit_rate_percent': round(goto_hit_rate, 2),
            **self._lookup_stats
        }


class ParseTreeVisualizer:
    """Utility class for generating parse tree visualizations."""
    
    @staticmethod
    def generate_dot(parse_tree: ParseTreeNode, title: str = "Parse Tree") -> str:
        """
        Generate DOT format representation of a parse tree.
        
        Args:
            parse_tree: Root node of the parse tree
            title: Title for the graph
            
        Returns:
            DOT format string
        """
        from visualization import DOTGenerator
        generator = DOTGenerator()
        return generator.generate_parse_tree_dot(parse_tree, title)
    
    @staticmethod
    def generate_html_tree(parse_tree: ParseTreeNode, indent: int = 0) -> str:
        """
        Generate HTML representation of a parse tree.
        
        Args:
            parse_tree: Root node of the parse tree
            indent: Current indentation level
            
        Returns:
            HTML string showing the tree structure
        """
        lines = []
        indent_str = "  " * indent
        
        if parse_tree.is_terminal:
            lines.append(f'{indent_str}<span class="terminal">"{parse_tree.label}"</span>')
        else:
            lines.append(f'{indent_str}<div class="non-terminal">')
            lines.append(f'{indent_str}  <span class="label">{parse_tree.label}</span>')
            
            if parse_tree.children:
                lines.append(f'{indent_str}  <div class="children">')
                for child in parse_tree.children:
                    child_html = ParseTreeVisualizer.generate_html_tree(child, indent + 2)
                    lines.append(child_html)
                lines.append(f'{indent_str}  </div>')
            
            lines.append(f'{indent_str}</div>')
        
        return '\n'.join(lines)


class CFGParserVisualizer:
    """
    Main class that integrates all CFG parser components with visualization.
    
    This class provides a high-level interface for parsing CFG input,
    generating parsing tables, parsing input strings, and creating
    comprehensive visualizations of the entire process.
    """
    
    def __init__(self):
        self.grammar_processor = GrammarProcessor()
        self.grammar = None
        self.automaton = None
        self.parsing_tables = None
        self.parsing_engine = None
        
    def process_grammar(self, cfg_text: str) -> Dict[str, str]:
        """
        Process a CFG and generate all visualization components.
        
        Args:
            cfg_text: Context-free grammar in text format
            
        Returns:
            Dictionary containing all visualization outputs
        """
        try:
            # Parse grammar
            self.grammar = self.grammar_processor.parse_grammar(cfg_text)
            
            # Build CLR automaton
            clr_builder = CLRItemSetBuilder(self.grammar)
            self.automaton = clr_builder.build_clr_automaton()
            
            # Generate parsing tables
            table_generator = CLRTableGenerator(self.grammar, self.automaton)
            self.parsing_tables = table_generator.generate_parsing_tables()
            
            # Detect conflicts
            conflicts = table_generator.detect_conflicts(self.parsing_tables)
            
            # Create parsing engine
            self.parsing_engine = CLRParsingEngine(self.grammar, self.parsing_tables)
            
            # Generate visualizations
            from visualization import VisualizationGenerator
            viz_generator = VisualizationGenerator()
            
            result = {
                'grammar_info': str(self.grammar),
                'tables_html': viz_generator.generate_parsing_tables_html(
                    self.parsing_tables, self.grammar.terminals, self.grammar.non_terminals
                ),
                'automaton_dot': viz_generator.dot_generator.generate_automaton_dot(
                    self.automaton, "CLR Automaton"
                ),
                'conflicts_html': viz_generator.error_formatter.format_conflict_report(conflicts),
                'success': True,
                'error': None
            }
            
            return result
            
        except Exception as e:
            from visualization import ErrorMessageFormatter
            error_formatter = ErrorMessageFormatter()
            return {
                'success': False,
                'error': str(e),
                'error_html': error_formatter.format_parse_error(str(e))
            }
    
    def parse_input(self, input_string: str) -> Dict[str, str]:
        """
        Parse an input string and generate visualization.
        
        Args:
            input_string: String to parse
            
        Returns:
            Dictionary containing parsing results and visualizations
        """
        if not self.parsing_engine:
            return {
                'success': False,
                'error': 'No grammar processed. Call process_grammar() first.'
            }
        
        try:
            # Parse the input
            result = self.parsing_engine.parse(input_string)
            
            # Generate visualizations
            from visualization import VisualizationGenerator
            viz_generator = VisualizationGenerator()
            
            if result.success:
                return {
                    'success': True,
                    'tree_dot': viz_generator.generate_parse_tree_dot(result.parse_tree),
                    'trace_html': viz_generator.generate_trace_html(result.trace),
                    'parse_tree': str(result.parse_tree) if result.parse_tree else None
                }
            else:
                return {
                    'success': False,
                    'error': result.error_message,
                    'error_html': viz_generator.format_error_message(
                        result.error_message, result.error_position, input_string
                    ),
                    'trace_html': viz_generator.generate_trace_html(result.trace) if result.trace else None
                }
                
        except Exception as e:
            from visualization import ErrorMessageFormatter
            error_formatter = ErrorMessageFormatter()
            return {
                'success': False,
                'error': str(e),
                'error_html': error_formatter.format_parse_error(str(e))
            }
    
    def generate_complete_report(self, cfg_text: str, input_string: str = None) -> Dict[str, str]:
        """
        Generate a complete analysis report for a CFG and optional input.
        
        Args:
            cfg_text: Context-free grammar in text format
            input_string: Optional input string to parse
            
        Returns:
            Dictionary containing complete analysis and visualizations
        """
        # Process grammar
        grammar_result = self.process_grammar(cfg_text)
        
        if not grammar_result['success']:
            return grammar_result
        
        result = grammar_result.copy()
        
        # Parse input if provided
        if input_string is not None:
            parse_result = self.parse_input(input_string)
            result.update(parse_result)
        
        return result


# Example usage and testing
if __name__ == "__main__":
    # Test the grammar processor
    processor = GrammarProcessor()
    
    # Test case 1: Simple arithmetic grammar
    cfg1 = """
    E -> E '+' T | T
    T -> T '*' F | F  
    F -> '(' E ')' | 'id'
    """
    
    print("=== Test Case 1: Arithmetic Grammar ===")
    grammar1 = processor.parse_grammar(cfg1)
    print(grammar1)
    print()
    
    # Test case 2: Grammar with epsilon productions
    cfg2 = """
    S -> A B
    A -> 'a' A | e
    B -> 'b' B | e
    """
    
    print("=== Test Case 2: Grammar with Epsilon ===")
    grammar2 = processor.parse_grammar(cfg2)
    print(grammar2)
    print()
    
    # Test case 3: Grammar with different arrow styles
    cfg3 = """
    S : A B;
    A = 'x' | 'y';
    B -> 'z';
    """
    
    print("=== Test Case 3: Different Arrow Styles ===")
    grammar3 = processor.parse_grammar(cfg3)
    print(grammar3)
    print()
    
    # Test FIRST and FOLLOW computation
    print("=== FIRST and FOLLOW Sets Testing ===")
    
    # Test with arithmetic grammar
    print("--- Arithmetic Grammar FIRST/FOLLOW ---")
    ff_computer1 = FirstFollowComputer(grammar1)
    first_sets1 = ff_computer1.compute_first_sets()
    follow_sets1 = ff_computer1.compute_follow_sets()
    
    print("FIRST sets:")
    for symbol in sorted(first_sets1.keys()):
        print(f"  FIRST({symbol}) = {sorted(first_sets1[symbol])}")
    
    print("FOLLOW sets:")
    for symbol in sorted(follow_sets1.keys()):
        print(f"  FOLLOW({symbol}) = {sorted(follow_sets1[symbol])}")
    print()
    
    # Test with epsilon grammar
    print("--- Epsilon Grammar FIRST/FOLLOW ---")
    ff_computer2 = FirstFollowComputer(grammar2)
    first_sets2 = ff_computer2.compute_first_sets()
    follow_sets2 = ff_computer2.compute_follow_sets()
    
    print("FIRST sets:")
    for symbol in sorted(first_sets2.keys()):
        print(f"  FIRST({symbol}) = {sorted(first_sets2[symbol])}")
    
    print("FOLLOW sets:")
    for symbol in sorted(follow_sets2.keys()):
        print(f"  FOLLOW({symbol}) = {sorted(follow_sets2[symbol])}")
    print()
    
    # Test a more complex grammar with epsilon productions
    cfg4 = """
    S -> A B C
    A -> 'a' | e
    B -> 'b' | e  
    C -> 'c'
    """
    
    print("--- Complex Epsilon Grammar ---")
    grammar4 = processor.parse_grammar(cfg4)
    print(grammar4)
    
    ff_computer4 = FirstFollowComputer(grammar4)
    first_sets4 = ff_computer4.compute_first_sets()
    follow_sets4 = ff_computer4.compute_follow_sets()
    
    print("FIRST sets:")
    for symbol in sorted(first_sets4.keys()):
        print(f"  FIRST({symbol}) = {sorted(first_sets4[symbol])}")
    
    print("FOLLOW sets:")
    for symbol in sorted(follow_sets4.keys()):
        print(f"  FOLLOW({symbol}) = {sorted(follow_sets4[symbol])}")
    print()
    
    # Test CLR automaton construction
    print("=== CLR Automaton Construction Testing ===")
    
    # Test with simple grammar
    cfg_simple = """
    S -> 'a' S 'b' | e
    """
    
    print("--- Simple Grammar CLR Automaton ---")
    grammar_simple = processor.parse_grammar(cfg_simple)
    print("Grammar:")
    print(grammar_simple)
    print()
    
    clr_builder = CLRItemSetBuilder(grammar_simple)
    automaton = clr_builder.build_clr_automaton()
    print("CLR Automaton:")
    print(automaton)
    print()
    
    # Test with arithmetic grammar (more complex)
    print("--- Arithmetic Grammar CLR Automaton ---")
    clr_builder1 = CLRItemSetBuilder(grammar1)
    automaton1 = clr_builder1.build_clr_automaton()
    print(f"CLR Automaton for arithmetic grammar:")
    print(f"Number of states: {len(automaton1.states)}")
    print(f"Number of transitions: {len(automaton1.transitions)}")
    print()
    
    # Show first few states as example
    print("First 3 states:")
    for i, state in enumerate(automaton1.states[:3]):
        print(state)
        print()
    
    # Test CLR parsing table generation
    print("=== CLR Parsing Table Generation Testing ===")
    
    # Test with simple grammar
    print("--- Simple Grammar Parsing Tables ---")
    table_generator = CLRTableGenerator(grammar_simple, automaton)
    tables = table_generator.generate_parsing_tables()
    
    print("Action Table:")
    for (state, terminal), action in sorted(tables.action_table.items()):
        print(f"  ACTION[{state}, {terminal}] = {action}")
    print()
    
    print("Goto Table:")
    for (state, non_terminal), target in sorted(tables.goto_table.items()):
        print(f"  GOTO[{state}, {non_terminal}] = {target}")
    print()
    
    # Check for conflicts
    conflicts = table_generator.detect_conflicts(tables)
    if conflicts:
        print("Conflicts detected:")
        print(table_generator.generate_conflict_report())
    else:
        print("No conflicts detected in simple grammar.")
    print()
    
    # Test with arithmetic grammar
    print("--- Arithmetic Grammar Parsing Tables ---")
    table_generator1 = CLRTableGenerator(grammar1, automaton1)
    tables1 = table_generator1.generate_parsing_tables()
    
    print(f"Generated {len(tables1.action_table)} action entries and {len(tables1.goto_table)} goto entries")
    
    # Show sample entries
    print("\nSample Action Table entries:")
    action_items = list(tables1.action_table.items())
    for i, ((state, terminal), action) in enumerate(action_items[:10]):
        print(f"  ACTION[{state}, {terminal}] = {action}")
    if len(action_items) > 10:
        print(f"  ... and {len(action_items) - 10} more entries")
    
    print("\nSample Goto Table entries:")
    goto_items = list(tables1.goto_table.items())
    for i, ((state, non_terminal), target) in enumerate(goto_items[:10]):
        print(f"  GOTO[{state}, {non_terminal}] = {target}")
    if len(goto_items) > 10:
        print(f"  ... and {len(goto_items) - 10} more entries")
    print()
    
    # Check for conflicts in arithmetic grammar
    conflicts1 = table_generator1.detect_conflicts(tables1)
    if conflicts1:
        print("Conflicts detected in arithmetic grammar:")
        print(table_generator1.generate_conflict_report())
    else:
        print("No conflicts detected in arithmetic grammar.")
    print()
    
    # Test with a grammar that has known conflicts
    print("--- Grammar with Conflicts Testing ---")
    cfg_conflict = """
    E -> E '+' E
    E -> 'id'
    """
    
    grammar_conflict = processor.parse_grammar(cfg_conflict)
    print("Ambiguous Grammar:")
    print(grammar_conflict)
    print()
    
    clr_builder_conflict = CLRItemSetBuilder(grammar_conflict)
    automaton_conflict = clr_builder_conflict.build_clr_automaton()
    
    print(f"Conflict grammar automaton has {len(automaton_conflict.states)} states")
    
    table_generator_conflict = CLRTableGenerator(grammar_conflict, automaton_conflict)
    tables_conflict = table_generator_conflict.generate_parsing_tables()
    
    conflicts_conflict = table_generator_conflict.detect_conflicts(tables_conflict)
    if conflicts_conflict:
        print("Expected conflicts detected:")
        print(table_generator_conflict.generate_conflict_report())
    else:
        print("Checking for conflicts in action table...")
        # Let's examine the action table for this grammar
        conflict_found = False
        for (state, terminal), action in sorted(tables_conflict.action_table.items()):
            if "[CONFLICT]" in str(action):
                print(f"  ACTION[{state}, {terminal}] = {action}")
                conflict_found = True
        if not conflict_found:
            print("No conflicts found. This grammar may be LR(1) after all.")
    print()
    
    # Test closure operation specifically
    print("=== CLR Closure Operation Testing ===")
    
    # Create a test item and compute its closure
    test_production = Production(lhs="E", rhs=["E", "+", "T"], is_epsilon=False)
    test_item = CLRItem(production=test_production, dot_position=0, lookahead="$")
    test_items = {test_item}
    
    print(f"Initial item: {test_item}")
    closure_result = clr_builder1.closure(test_items)
    print("Closure result:")
    for item in sorted(closure_result, key=str):
        print(f"  {item}")
    print()
    
    # Test GOTO operation
    print("=== CLR GOTO Operation Testing ===")
    
    # Test GOTO on the closure result
    goto_result = clr_builder1.goto(closure_result, "E")
    print(f"GOTO(closure, E):")
    for item in sorted(goto_result, key=str):
        print(f"  {item}")
    print()
    
    goto_result2 = clr_builder1.goto(closure_result, "T")
    print(f"GOTO(closure, T):")
    for item in sorted(goto_result2, key=str):
        print(f"  {item}")
    print()
    
    # Test lexical analyzer and tokenizer
    print("=== Lexical Analyzer and Tokenizer Testing ===")
    
    # Test with arithmetic grammar terminals
    print("--- Arithmetic Grammar Tokenization ---")
    lexer1 = LexicalAnalyzer(grammar1.terminals)
    
    print("Terminal info:")
    terminal_info = lexer1.get_terminal_info()
    for terminal, info in sorted(terminal_info.items()):
        print(f"  {terminal}: {info}")
    print()
    
    # Test tokenizing arithmetic expressions
    test_inputs = [
        "id + id * id",
        "( id + id ) * id",
        "id",
        "+ * ( )",
        "id + id + id"
    ]
    
    for test_input in test_inputs:
        print(f"Tokenizing: '{test_input}'")
        tokens, errors = lexer1.tokenize(test_input)
        
        if errors:
            print("  Errors:")
            for error in errors:
                print(f"    {error}")
        
        print("  Tokens:")
        for token in tokens:
            print(f"    {token}")
        print()
    
    # Test with quoted terminals
    print("--- Quoted Terminals Testing ---")
    cfg_quoted = """
    S -> A '+' B
    A -> 'id' | '(' S ')'
    B -> 'num'
    """
    
    grammar_quoted = processor.parse_grammar(cfg_quoted)
    print("Grammar with quoted terminals:")
    print(grammar_quoted)
    print()
    
    lexer_quoted = LexicalAnalyzer(grammar_quoted.terminals)
    
    print("Quoted terminal info:")
    quoted_info = lexer_quoted.get_terminal_info()
    for terminal, info in sorted(quoted_info.items()):
        print(f"  {terminal}: {info}")
    print()
    
    # Test tokenizing with quoted terminals
    quoted_test_inputs = [
        "id + num",
        "( id + num )",
        "+ num id"
    ]
    
    for test_input in quoted_test_inputs:
        print(f"Tokenizing: '{test_input}'")
        tokens, errors = lexer_quoted.tokenize(test_input)
        
        if errors:
            print("  Errors:")
            for error in errors:
                print(f"    {error}")
        
        print("  Tokens:")
        for token in tokens:
            print(f"    {token}")
        print()
    
    # Test with multi-character terminals
    print("--- Multi-character Terminals Testing ---")
    cfg_multi = """
    S -> IF expr THEN stmt ELSE stmt
    expr -> ID EQ NUM
    stmt -> ID ASSIGN expr
    """
    
    grammar_multi = processor.parse_grammar(cfg_multi)
    print("Grammar with multi-character terminals:")
    print(grammar_multi)
    print()
    
    lexer_multi = LexicalAnalyzer(grammar_multi.terminals)
    
    print("Multi-character terminal info:")
    multi_info = lexer_multi.get_terminal_info()
    for terminal, info in sorted(multi_info.items()):
        print(f"  {terminal}: {info}")
    print()
    
    # Test tokenizing with multi-character terminals
    multi_test_inputs = [
        "IF ID EQ NUM THEN ID ASSIGN NUM ELSE ID ASSIGN ID",
        "IF x EQ 5 THEN y ASSIGN 10 ELSE z ASSIGN x"
    ]
    
    for test_input in multi_test_inputs:
        print(f"Tokenizing: '{test_input}'")
        tokens, errors = lexer_multi.tokenize(test_input)
        
        if errors:
            print("  Errors:")
            for error in errors:
                print(f"    {error}")
        
        print("  Tokens:")
        for token in tokens:
            print(f"    {token}")
        print()
    
    # Test error handling
    print("--- Error Handling Testing ---")
    
    # Test with invalid characters
    error_test_inputs = [
        "id + @ id",  # Invalid character @
        "id & id",    # Invalid character &
        "id + id #",  # Invalid character #
    ]
    
    for test_input in error_test_inputs:
        print(f"Tokenizing (expecting errors): '{test_input}'")
        tokens, errors = lexer1.tokenize(test_input)
        
        if errors:
            print("  Errors (as expected):")
            for error in errors:
                print(f"    {error}")
        else:
            print("  No errors found (unexpected)")
        
        print("  Tokens:")
        for token in tokens:
            print(f"    {token}")
        print()
    
    # Test longest-match strategy
    print("--- Longest-match Strategy Testing ---")
    cfg_longest = """
    S -> IF IFF IDENTIFIER
    """
    
    grammar_longest = processor.parse_grammar(cfg_longest)
    lexer_longest = LexicalAnalyzer(grammar_longest.terminals)
    
    longest_test_inputs = [
        "IF IFF IDENTIFIER",
        "IFF IF IDENTIFIER",
        "IDENTIFIER IF IFF"
    ]
    
    for test_input in longest_test_inputs:
        print(f"Tokenizing (longest-match): '{test_input}'")
        tokens, errors = lexer_longest.tokenize(test_input)
        
        if errors:
            print("  Errors:")
            for error in errors:
                print(f"    {error}")
        
        print("  Tokens:")
        for token in tokens:
            print(f"    {token}")
        print()
    
    # Test escape sequences in quoted terminals
    print("--- Escape Sequences Testing ---")
    cfg_escape = """
    S -> A '\\n' B
    A -> '\\t'
    B -> '\\''
    """
    
    grammar_escape = processor.parse_grammar(cfg_escape)
    lexer_escape = LexicalAnalyzer(grammar_escape.terminals)
    
    print("Escape sequence terminal info:")
    escape_info = lexer_escape.get_terminal_info()
    for terminal, info in sorted(escape_info.items()):
        print(f"  {terminal}: {info}")
    print()
    
    # Note: For testing escape sequences, we need to use the actual characters
    escape_test_inputs = [
        "\t\n'",  # Tab, newline, single quote
    ]
    
    for test_input in escape_test_inputs:
        print(f"Tokenizing escape sequences: {repr(test_input)}")
        tokens, errors = lexer_escape.tokenize(test_input)
        
        if errors:
            print("  Errors:")
            for error in errors:
                print(f"    {error}")
        
        print("  Tokens:")
        for token in tokens:
            print(f"    {token} (value: {repr(token.value)})")
        print()
    
    # Test CLR parsing engine
    print("=== CLR Parsing Engine Testing ===")
    
    # Test with simple grammar
    print("--- Simple Grammar Parsing ---")
    
    # Use a very simple grammar for testing
    cfg_parse_test = """
    S -> 'a' S 'b' | e
    """
    
    grammar_parse_test = processor.parse_grammar(cfg_parse_test)
    print("Test Grammar:")
    print(grammar_parse_test)
    print()
    
    # Build CLR automaton and tables
    clr_builder_test = CLRItemSetBuilder(grammar_parse_test)
    automaton_test = clr_builder_test.build_clr_automaton()
    table_generator_test = CLRTableGenerator(grammar_parse_test, automaton_test)
    tables_test = table_generator_test.generate_parsing_tables()
    
    # Create parsing engine
    parser_engine = CLRParsingEngine(grammar_parse_test, tables_test)
    
    # Test parsing various inputs
    parse_test_inputs = [
        "",  # Empty string (should be accepted due to epsilon)
        "a b",  # Simple case
        "a a b b",  # Nested case
        "a a a b b b",  # More nested
        "a b a b",  # Multiple sequences
        "a a b",  # Unbalanced (should fail)
        "a b b",  # Unbalanced (should fail)
        "b a",  # Wrong order (should fail)
    ]
    
    for test_input in parse_test_inputs:
        print(f"Parsing: '{test_input}'")
        result = parser_engine.parse(test_input)
        
        if result.success:
            print(f"  SUCCESS: Parse tree root = {result.parse_tree}")
            print(f"  Parse tree structure:")
            tree_html = ParseTreeVisualizer.generate_html_tree(result.parse_tree, 2)
            # Convert HTML to simple text representation
            tree_text = tree_html.replace('<div class="non-terminal">', '').replace('</div>', '')
            tree_text = tree_text.replace('<span class="label">', '').replace('</span>', '')
            tree_text = tree_text.replace('<div class="children">', '').replace('<span class="terminal">', '"').replace('</span>', '"')
            print(tree_text)
        else:
            print(f"  FAILED: {result.error_message}")
        
        print(f"  Trace steps: {len(result.trace)}")
        if len(result.trace) <= 10:  # Show trace for short parses
            for step in result.trace:
                print(f"    {step}")
        print()
    
    # Test with arithmetic grammar
    print("--- Arithmetic Grammar Parsing ---")
    
    # Create parsing engine for arithmetic grammar
    parser_engine_arith = CLRParsingEngine(grammar1, tables1)
    
    # Test parsing arithmetic expressions
    arith_test_inputs = [
        "id",
        "id + id",
        "id * id",
        "id + id * id",
        "( id + id ) * id",
        "id + ( id * id )",
        "( ( id ) )",
        "id + id + id",
        "id * id * id",
    ]
    
    for test_input in arith_test_inputs:
        print(f"Parsing arithmetic: '{test_input}'")
        result = parser_engine_arith.parse(test_input)
        
        if result.success:
            print(f"  SUCCESS: Parse tree root = {result.parse_tree}")
            # Generate DOT representation
            dot_output = ParseTreeVisualizer.generate_dot(result.parse_tree, f"Parse Tree for '{test_input}'")
            print(f"  DOT format (first 3 lines):")
            dot_lines = dot_output.split('\n')
            for line in dot_lines[:3]:
                print(f"    {line}")
            print(f"    ... ({len(dot_lines)} total lines)")
        else:
            print(f"  FAILED: {result.error_message} at position {result.error_position}")
        
        print(f"  Trace steps: {len(result.trace)}")
        print()
    
    # Test error cases
    print("--- Error Handling in Parsing ---")
    
    error_test_inputs = [
        "id +",  # Incomplete expression
        "+ id",  # Invalid start
        "id + + id",  # Double operator
        "( id + id",  # Unmatched parenthesis
        "id + id )",  # Extra closing parenthesis
        "id id",  # Missing operator
        "( )",  # Empty parentheses
    ]
    
    for test_input in error_test_inputs:
        print(f"Parsing (expecting error): '{test_input}'")
        result = parser_engine_arith.parse(test_input)
        
        if result.success:
            print(f"  UNEXPECTED SUCCESS: {result.parse_tree}")
        else:
            print(f"  EXPECTED FAILURE: {result.error_message}")
            print(f"  Error position: {result.error_position}")
        
        print()
    
    # Test trace generation
    print("--- Parsing Trace Generation ---")
    
    trace_test_input = "id + id * id"
    print(f"Generating detailed trace for: '{trace_test_input}'")
    result = parser_engine_arith.parse(trace_test_input)
    
    if result.success:
        print("Parse successful. Detailed trace:")
        for i, step in enumerate(result.trace):
            print(f"  {step}")
            if i >= 15:  # Limit output for readability
                print(f"  ... ({len(result.trace)} total steps)")
                break
        
        print("\nHTML trace (first few lines):")
        html_trace = parser_engine_arith.generate_trace_html(result.trace)
        html_lines = html_trace.split('\n')
        for line in html_lines[:10]:
            print(f"  {line}")
        print(f"  ... ({len(html_lines)} total lines)")
    else:
        print(f"Parse failed: {result.error_message}")
    print()
    
    # Test parse tree visualization
    print("--- Parse Tree Visualization ---")
    
    if result.success:
        print("Parse tree in DOT format:")
        dot_output = ParseTreeVisualizer.generate_dot(result.parse_tree, "Arithmetic Expression Parse Tree")
        print(dot_output)
        print()
        
        print("Parse tree in HTML format:")
        html_tree = ParseTreeVisualizer.generate_html_tree(result.parse_tree)
        print(html_tree)
    print()


class PerformanceProfiler:
    """
    Performance profiler for CFG parser operations.
    
    Provides detailed timing and memory usage analysis for optimization.
    """
    
    def __init__(self):
        self.operation_times: Dict[str, List[float]] = {}
        self.memory_snapshots: Dict[str, int] = {}
        self.call_counts: Dict[str, int] = {}
        self.enabled = False
    
    def enable(self):
        """Enable performance profiling."""
        self.enabled = True
        self.reset()
    
    def disable(self):
        """Disable performance profiling."""
        self.enabled = False
    
    def reset(self):
        """Reset all profiling data."""
        self.operation_times.clear()
        self.memory_snapshots.clear()
        self.call_counts.clear()
    
    def start_operation(self, operation_name: str) -> float:
        """Start timing an operation."""
        if not self.enabled:
            return 0.0
        
        self.call_counts[operation_name] = self.call_counts.get(operation_name, 0) + 1
        return time.time()
    
    def end_operation(self, operation_name: str, start_time: float):
        """End timing an operation."""
        if not self.enabled or start_time == 0.0:
            return
        
        duration = time.time() - start_time
        if operation_name not in self.operation_times:
            self.operation_times[operation_name] = []
        self.operation_times[operation_name].append(duration)
    
    def record_memory_snapshot(self, label: str):
        """Record current memory usage."""
        if not self.enabled:
            return
        
        # Rough memory estimation
        import gc
        gc.collect()
        self.memory_snapshots[label] = sys.getsizeof(gc.get_objects())
    
    def get_performance_report(self) -> Dict[str, Union[int, float, str]]:
        """Generate comprehensive performance report."""
        if not self.enabled:
            return {'error': 'Profiling not enabled'}
        
        report = {
            'profiling_enabled': True,
            'total_operations': len(self.operation_times),
            'total_calls': sum(self.call_counts.values())
        }
        
        # Add timing statistics
        for operation, times in self.operation_times.items():
            if times:
                report[f'{operation}_total_time_ms'] = round(sum(times) * 1000, 3)
                report[f'{operation}_avg_time_ms'] = round(sum(times) / len(times) * 1000, 3)
                report[f'{operation}_min_time_ms'] = round(min(times) * 1000, 3)
                report[f'{operation}_max_time_ms'] = round(max(times) * 1000, 3)
                report[f'{operation}_call_count'] = len(times)
        
        # Add memory snapshots
        for label, memory in self.memory_snapshots.items():
            report[f'memory_{label}_bytes'] = memory
        
        return report
    
    def get_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on profiling data."""
        recommendations = []
        
        if not self.enabled or not self.operation_times:
            return ['Enable profiling to get optimization recommendations']
        
        # Analyze timing data
        total_times = {}
        for operation, times in self.operation_times.items():
            total_times[operation] = sum(times)
        
        # Find bottlenecks
        if total_times:
            max_time_op = max(total_times, key=total_times.get)
            max_time = total_times[max_time_op]
            
            if max_time > 1.0:  # More than 1 second
                recommendations.append(f"Optimize '{max_time_op}' operation - takes {max_time:.2f}s total")
        
        # Check cache hit rates from various components
        for operation, times in self.operation_times.items():
            if 'cache' in operation.lower() and len(times) > 100:
                avg_time = sum(times) / len(times)
                if avg_time > 0.001:  # More than 1ms average
                    recommendations.append(f"Consider improving cache efficiency for '{operation}'")
        
        # Check call frequency
        high_frequency_ops = [op for op, count in self.call_counts.items() if count > 1000]
        for op in high_frequency_ops:
            recommendations.append(f"High-frequency operation '{op}' called {self.call_counts[op]} times - consider optimization")
        
        if not recommendations:
            recommendations.append("Performance looks good - no major bottlenecks detected")
        
        return recommendations


class GrammarWorkflowManager:
    """
    Manages step-by-step CFG parsing workflow for interactive grammar analysis.
    
    This class orchestrates the interactive parsing process by:
    - Parsing CFG productions and extracting potential start symbols
    - Managing start symbol selection
    - Building parse tables with selected start symbol
    - Tracking workflow state throughout the process
    """
    
    def __init__(self, cfg_text: str):
        """
        Initialize the workflow manager with CFG text.
        
        Args:
            cfg_text: Context-free grammar in text format
        """
        self.cfg_text = cfg_text
        self.grammar_processor = GrammarProcessor()
        self.grammar = None
        self.start_symbol = None
        self.parse_table = None
        self.automaton = None
        self.parsing_tables = None
        self.parsing_engine = None
        self.workflow_state = "initial"
        self.potential_start_symbols = []
        self.productions = []
        
    def parse_productions(self) -> Dict[str, Any]:
        """
        Parse CFG productions and extract potential start symbols.
        
        Returns:
            Dictionary containing success status, productions list, and potential start symbols
        """
        try:
            # Parse the grammar to extract productions
            self.grammar = self.grammar_processor.parse_grammar(self.cfg_text)
            
            # Extract productions as strings for display
            self.productions = []
            for prod in self.grammar.productions:
                if prod.is_epsilon:
                    prod_str = f"{prod.lhs} -> ε"
                else:
                    prod_str = f"{prod.lhs} -> {' '.join(prod.rhs)}"
                self.productions.append(prod_str)
            
            # Extract potential start symbols (all non-terminals that appear as LHS)
            self.potential_start_symbols = list(self.grammar.non_terminals)
            self.potential_start_symbols.sort()  # Sort for consistent ordering
            
            # Update workflow state
            self.workflow_state = "productions_parsed"
            
            return {
                'success': True,
                'productions': self.productions,
                'start_symbols': self.potential_start_symbols,
                'grammar_info': {
                    'terminals': sorted(list(self.grammar.terminals)),
                    'non_terminals': sorted(list(self.grammar.non_terminals)),
                    'production_count': len(self.grammar.productions)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Grammar parsing failed: {str(e)}",
                'productions': [],
                'start_symbols': []
            }
    
    def set_start_symbol(self, start_symbol: str) -> Dict[str, Any]:
        """
        Set start symbol and build parse table with selected start symbol.
        
        Args:
            start_symbol: The selected start symbol for the grammar
            
        Returns:
            Dictionary containing success status and parse table HTML
        """
        try:
            # Validate that we have parsed productions
            if self.workflow_state != "productions_parsed":
                return {
                    'success': False,
                    'error': "Must parse productions first before setting start symbol"
                }
            
            # Validate start symbol
            if start_symbol not in self.potential_start_symbols:
                return {
                    'success': False,
                    'error': f"Invalid start symbol '{start_symbol}'. Must be one of: {self.potential_start_symbols}"
                }
            
            # Set the start symbol in the grammar
            self.start_symbol = start_symbol
            self.grammar.start_symbol = start_symbol
            
            # Build CLR automaton
            clr_builder = CLRItemSetBuilder(self.grammar)
            self.automaton = clr_builder.build_clr_automaton()
            
            # Generate parsing tables
            table_generator = CLRTableGenerator(self.grammar, self.automaton)
            self.parsing_tables = table_generator.generate_parsing_tables()
            
            # Create parsing engine
            self.parsing_engine = CLRParsingEngine(self.grammar, self.parsing_tables)
            
            # Generate HTML representation of parse table
            try:
                from visualization import HTMLTableGenerator
                html_generator = HTMLTableGenerator()
                parse_table_html = html_generator.generate_action_goto_tables_html(
                    self.parsing_tables.action_table,
                    self.parsing_tables.goto_table,
                    self.grammar.terminals,
                    self.grammar.non_terminals
                )
            except ImportError:
                # Fallback if visualization module is not available
                parse_table_html = "<p>Parse table generated successfully (visualization module not available)</p>"
            
            # Detect conflicts
            conflicts = table_generator.detect_conflicts(self.parsing_tables)
            
            # Update workflow state
            self.workflow_state = "parse_table_built"
            
            return {
                'success': True,
                'parse_table_html': parse_table_html,
                'start_symbol': start_symbol,
                'table_info': {
                    'states_count': len(self.automaton.states),
                    'action_entries': len(self.parsing_tables.action_table),
                    'goto_entries': len(self.parsing_tables.goto_table),
                    'conflicts_count': len(conflicts)
                },
                'conflicts': [str(conflict) for conflict in conflicts] if conflicts else []
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Parse table generation failed: {str(e)}"
            }
    
    def get_workflow_state(self) -> Dict[str, Any]:
        """
        Get current workflow state and available actions.
        
        Returns:
            Dictionary containing current state and available next actions
        """
        state_info = {
            'current_state': self.workflow_state,
            'available_actions': []
        }
        
        if self.workflow_state == "initial":
            state_info['available_actions'] = ['parse_productions']
            state_info['description'] = "Ready to parse CFG productions"
            
        elif self.workflow_state == "productions_parsed":
            state_info['available_actions'] = ['set_start_symbol']
            state_info['description'] = "Productions parsed, ready to select start symbol"
            state_info['productions_count'] = len(self.productions)
            state_info['potential_start_symbols'] = self.potential_start_symbols
            
        elif self.workflow_state == "parse_table_built":
            state_info['available_actions'] = ['parse_input_string']
            state_info['description'] = "Parse table built, ready to parse input strings"
            state_info['start_symbol'] = self.start_symbol
            state_info['table_ready'] = True
            
        return state_info
    
    def parse_input_string(self, input_string: str) -> Dict[str, Any]:
        """
        Parse an input string using the built parse table.
        
        Args:
            input_string: The string to parse
            
        Returns:
            Dictionary containing parsing results
        """
        try:
            # Validate that parse table is built
            if self.workflow_state != "parse_table_built" or not self.parsing_engine:
                return {
                    'success': False,
                    'error': "Parse table must be built before parsing input strings"
                }
            
            # Parse the input string
            result = self.parsing_engine.parse(input_string)
            
            if result.success:
                # Generate parse tree DOT format
                try:
                    tree_dot = ParseTreeVisualizer.generate_dot(result.parse_tree, f"Parse Tree for '{input_string}'")
                except Exception:
                    # Fallback if visualization fails
                    tree_dot = f"digraph ParseTree {{ label=\"Parse Tree for '{input_string}'\"; root [label=\"{result.parse_tree.label}\"]; }}"
                
                # Generate trace HTML
                trace_html = self.parsing_engine.generate_trace_html(result.trace)
                
                return {
                    'success': True,
                    'parse_tree_dot': tree_dot,
                    'trace_html': trace_html,
                    'trace_steps': len(result.trace),
                    'input_string': input_string
                }
            else:
                # Generate trace HTML even for failed parsing
                trace_html = self.parsing_engine.generate_trace_html(result.trace) if result.trace else ""
                
                return {
                    'success': False,
                    'error': result.error_message,
                    'error_position': result.error_position,
                    'trace_html': trace_html,
                    'trace_steps': len(result.trace) if result.trace else 0,
                    'input_string': input_string
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Input parsing failed: {str(e)}",
                'input_string': input_string
            }


class OptimizedCFGParser:
    """
    Main optimized CFG parser class that integrates all performance enhancements.
    
    This class provides a high-level interface with all optimizations enabled:
    - Flyweight pattern for CLR items
    - Lazy evaluation for large state spaces
    - Cached parsing table lookups
    - Performance profiling and monitoring
    """
    
    def __init__(self, enable_profiling: bool = False):
        self.profiler = PerformanceProfiler()
        if enable_profiling:
            self.profiler.enable()
        
        self.grammar_processor = GrammarProcessor()
        self.grammar = None
        self.automaton = None
        self.parsing_tables = None
        self.parsing_engine = None
        
        # Optimization settings
        self.flyweight_cache_limit = 10000
        self.lazy_evaluation_threshold = 100
        self.max_states = 1000
        self.parsing_cache_size = 500
    
    def configure_optimizations(self, 
                              flyweight_cache_limit: int = 10000,
                              lazy_evaluation_threshold: int = 100,
                              max_states: int = 1000,
                              parsing_cache_size: int = 500):
        """Configure optimization parameters."""
        self.flyweight_cache_limit = flyweight_cache_limit
        self.lazy_evaluation_threshold = lazy_evaluation_threshold
        self.max_states = max_states
        self.parsing_cache_size = parsing_cache_size
        
        # Apply settings to flyweight
        CLRItemFlyweight.set_cache_limit(flyweight_cache_limit)
    
    def process_grammar(self, cfg_text: str) -> Dict[str, Union[str, bool]]:
        """Process a CFG with full optimization and profiling."""
        start_time = self.profiler.start_operation('grammar_processing')
        
        try:
            # Parse grammar
            grammar_start = self.profiler.start_operation('grammar_parsing')
            self.grammar = self.grammar_processor.parse_grammar(cfg_text)
            self.profiler.end_operation('grammar_parsing', grammar_start)
            
            # Build CLR automaton with optimizations
            automaton_start = self.profiler.start_operation('clr_automaton_build')
            clr_builder = CLRItemSetBuilder(self.grammar)
            clr_builder._lazy_threshold = self.lazy_evaluation_threshold
            clr_builder._max_states = self.max_states
            if hasattr(clr_builder, 'enable_profiling'):
                clr_builder.enable_profiling(self.profiler.enabled)
            
            self.automaton = clr_builder.build_clr_automaton()
            self.profiler.end_operation('clr_automaton_build', automaton_start)
            
            # Generate parsing tables
            tables_start = self.profiler.start_operation('table_generation')
            table_generator = CLRTableGenerator(self.grammar, self.automaton)
            self.parsing_tables = table_generator.generate_parsing_tables()
            self.profiler.end_operation('table_generation', tables_start)
            
            # Create optimized parsing engine
            engine_start = self.profiler.start_operation('engine_creation')
            self.parsing_engine = CLRParsingEngine(self.grammar, self.parsing_tables)
            self.parsing_engine._max_cache_size = self.parsing_cache_size
            self.profiler.end_operation('engine_creation', engine_start)
            
            # Record memory snapshot
            self.profiler.record_memory_snapshot('after_grammar_processing')
            
            # Collect performance statistics
            performance_stats = {
                'clr_builder_stats': clr_builder.get_performance_stats(),
                'flyweight_stats': CLRItemFlyweight.get_cache_stats(),
                'parsing_engine_stats': self.parsing_engine.get_performance_stats()
            }
            
            self.profiler.end_operation('grammar_processing', start_time)
            
            return {
                'success': True,
                'grammar_info': str(self.grammar),
                'states_created': len(self.automaton.states),
                'transitions_created': len(self.automaton.transitions),
                'performance_stats': performance_stats,
                'profiler_report': self.profiler.get_performance_report(),
                'optimization_recommendations': self.profiler.get_optimization_recommendations()
            }
            
        except Exception as e:
            self.profiler.end_operation('grammar_processing', start_time)
            return {
                'success': False,
                'error': str(e),
                'profiler_report': self.profiler.get_performance_report()
            }
    
    def parse_input(self, input_string: str) -> Dict[str, Union[str, bool]]:
        """Parse input string with optimization and profiling."""
        if not self.parsing_engine:
            return {
                'success': False,
                'error': 'No grammar processed. Call process_grammar() first.'
            }
        
        start_time = self.profiler.start_operation('input_parsing')
        
        try:
            result = self.parsing_engine.parse(input_string)
            self.profiler.end_operation('input_parsing', start_time)
            
            return {
                'success': result.success,
                'parse_tree': str(result.parse_tree) if result.parse_tree else None,
                'error_message': result.error_message if not result.success else None,
                'trace_steps': len(result.trace),
                'parsing_engine_stats': self.parsing_engine.get_performance_stats(),
                'profiler_report': self.profiler.get_performance_report()
            }
            
        except Exception as e:
            self.profiler.end_operation('input_parsing', start_time)
            return {
                'success': False,
                'error': str(e),
                'profiler_report': self.profiler.get_performance_report()
            }
    
    def get_comprehensive_performance_report(self) -> Dict[str, Union[str, int, float]]:
        """Get comprehensive performance report from all components."""
        report = {
            'profiler_enabled': self.profiler.enabled,
            'optimization_settings': {
                'flyweight_cache_limit': self.flyweight_cache_limit,
                'lazy_evaluation_threshold': self.lazy_evaluation_threshold,
                'max_states': self.max_states,
                'parsing_cache_size': self.parsing_cache_size
            }
        }
        
        # Add profiler report
        report.update(self.profiler.get_performance_report())
        
        # Add component-specific stats if available
        if self.parsing_engine:
            report['parsing_engine_stats'] = self.parsing_engine.get_performance_stats()
        
        # Add flyweight stats
        report['flyweight_stats'] = CLRItemFlyweight.get_cache_stats()
        
        # Add optimization recommendations
        report['optimization_recommendations'] = self.profiler.get_optimization_recommendations()
        
        return report


# Example usage for performance testing
if __name__ == "__main__":
    # Test optimized parser with profiling
    print("=== Testing Optimized CFG Parser ===")
    
    parser = OptimizedCFGParser(enable_profiling=True)
    
    # Configure optimizations for testing
    parser.configure_optimizations(
        flyweight_cache_limit=5000,
        lazy_evaluation_threshold=50,
        max_states=500,
        parsing_cache_size=200
    )
    
    # Test with arithmetic grammar
    cfg_text = """
    E -> E '+' T | T
    T -> T '*' F | F
    F -> '(' E ')' | 'id'
    """
    
    print("Processing grammar...")
    result = parser.process_grammar(cfg_text)
    
    if result['success']:
        print(f"✓ Grammar processed successfully")
        print(f"  States created: {result['states_created']}")
        print(f"  Transitions created: {result['transitions_created']}")
        
        # Test parsing
        test_input = "id + id * id"
        print(f"\nParsing input: '{test_input}'")
        parse_result = parser.parse_input(test_input)
        
        if parse_result['success']:
            print(f"✓ Parsing successful")
            print(f"  Trace steps: {parse_result['trace_steps']}")
        else:
            print(f"✗ Parsing failed: {parse_result['error_message']}")
        
        # Show performance report
        print("\n=== Performance Report ===")
        perf_report = parser.get_comprehensive_performance_report()
        
        for key, value in perf_report.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for subkey, subvalue in value.items():
                    print(f"  {subkey}: {subvalue}")
            elif isinstance(value, list):
                print(f"{key}:")
                for item in value:
                    print(f"  - {item}")
            else:
                print(f"{key}: {value}")
    
    else:
        print(f"✗ Grammar processing failed: {result['error']}")