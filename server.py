import subprocess
import re
import os
import sys
import json
import tempfile
import shutil
import traceback
from flask import Flask, request, jsonify, send_from_directory, abort

app = Flask(__name__)

# --- Define ABSOLUTE file paths ---
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REGEX_EXE_PATH = os.path.join(PROJECT_DIR, 'regex_visualizer.exe')

# --- Helper Function to check executables ---
def check_executables():
    missing = []
    if not os.path.exists(REGEX_EXE_PATH): 
        missing.append(REGEX_EXE_PATH)
    if missing:
        print("="*50, file=sys.stderr)
        print("! ! ! WARNING ! ! !", file=sys.stderr)
        missing_files_str = ", ".join(missing)
        print(f"Missing regex visualizer: {missing_files_str}", file=sys.stderr)
        print("Regex functionality will be unavailable. Run 'make all' to build regex_visualizer.exe.", file=sys.stderr)
        print("="*50, file=sys.stderr)
        return False
    return True

# --- HTML Escape Helper ---
def escapeHtml(unsafe):
    if unsafe is None: return ''
    unsafe = str(unsafe)
    return unsafe.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#039;')

# --- Flask Endpoints ---

@app.route('/')
def index():
    """Serves the main HTML file."""
    response = send_from_directory('.', 'index.html')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/visualize-dfa', methods=['POST'])
def visualize_dfa():
    """Visualize DFA using the regex visualizer executable."""
    data = request.json
    regex_input = data.get('regex')
    if not regex_input: 
        return jsonify({"error": "No regex provided"}), 400
    
    print(f"--- Attempting to run: {REGEX_EXE_PATH} ---", file=sys.stderr)
    try:
        process = subprocess.run([REGEX_EXE_PATH], input=regex_input, capture_output=True, text=True, timeout=10, check=False)
        if process.stderr: 
            print("--- Stderr from regex_visualizer.exe ---", file=sys.stderr)
            print(process.stderr, file=sys.stderr)
            print("-" * 42, file=sys.stderr)
        
        if process.returncode != 0:
            print(f"--- regex_visualizer.exe FAILED code {process.returncode} ---", file=sys.stderr)
            try: 
                error_json = json.loads(process.stdout)
                return jsonify({"error": error_json.get("error", f"Regex fail code {process.returncode}")}), 500
            except json.JSONDecodeError: 
                return jsonify({"error": process.stderr or f"Regex fail code {process.returncode}. No JSON."}), 500
        
        print("--- regex_visualizer.exe SUCCEEDED ---", file=sys.stderr)
        try: 
            output_data = json.loads(process.stdout)
            return jsonify(output_data)
        except json.JSONDecodeError as e: 
            print(f"--- ERROR: Bad JSON from regex_visualizer: {e} ---", file=sys.stderr)
            print("--- Raw stdout: ---\n", process.stdout, "\n", "-"*19, file=sys.stderr)
            return jsonify({"error": f"Backend JSON error: {e}"}), 500
    except FileNotFoundError: 
        print(f"--- FATAL ERROR: {REGEX_EXE_PATH} not found! ---", file=sys.stderr)
        return jsonify({"error": f"{REGEX_EXE_PATH} not found. Run 'make all'."}), 500
    except subprocess.TimeoutExpired: 
        print(f"--- TIMEOUT: {REGEX_EXE_PATH} ---", file=sys.stderr)
        return jsonify({"error": "Regex processing timed out."}), 500
    except Exception as e: 
        print(f"--- Python Error: {e} ---", file=sys.stderr)
        return jsonify({"error": str(e)}), 500

@app.route('/parse-cfg-productions', methods=['POST'])
def parse_cfg_productions():
    """
    Parse CFG input and return list of productions and potential start symbols.
    
    This endpoint is part of the interactive CFG workflow that allows users
    to see productions before selecting a start symbol.
    """
    data = request.json
    cfg_input = data.get('cfg')
    
    if not cfg_input:
        return jsonify({"error": "No CFG provided"}), 400

    try:
        # Import the workflow manager
        from cfg_parser import GrammarWorkflowManager
        
        print("--- Parsing CFG Productions ---", file=sys.stderr)
        
        # Create workflow manager
        workflow_manager = GrammarWorkflowManager(cfg_input)
        
        # Parse productions
        result = workflow_manager.parse_productions()
        
        if result['success']:
            print("--- Production Parsing SUCCEEDED ---", file=sys.stderr)
            print(f"Found {len(result['productions'])} productions", file=sys.stderr)
            print(f"Potential start symbols: {result['start_symbols']}", file=sys.stderr)
            
            return jsonify({
                "success": True,
                "productions": result['productions'],
                "start_symbols": result['start_symbols'],
                "grammar_info": result['grammar_info']
            })
        else:
            print(f"--- Production Parsing FAILED ---", file=sys.stderr)
            print(f"Error: {result['error']}", file=sys.stderr)
            
            return jsonify({
                "success": False,
                "error": result['error']
            }), 400
            
    except ImportError as e:
        print(f"--- IMPORT ERROR: Workflow manager not found: {e} ---", file=sys.stderr)
        return jsonify({"error": f"Workflow manager not available: {e}"}), 500
        
    except Exception as e:
        print(f"--- UNEXPECTED Python Error: {e} ---", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        error_message = f"Unexpected server error: {escapeHtml(str(e))}"
        return jsonify({"error": error_message}), 500

@app.route('/build-parse-table', methods=['POST'])
def build_parse_table():
    """
    Build parse table with CFG text and selected start symbol.
    
    This endpoint accepts CFG text and a selected start symbol, then generates
    the CLR parse table and returns it in HTML format for visualization.
    """
    data = request.json
    cfg_input = data.get('cfg')
    start_symbol = data.get('start_symbol')
    
    if not cfg_input:
        return jsonify({"error": "No CFG provided"}), 400
    if not start_symbol:
        return jsonify({"error": "No start symbol provided"}), 400

    try:
        # Import the workflow manager
        from cfg_parser import GrammarWorkflowManager
        
        print("--- Building Parse Table ---", file=sys.stderr)
        print(f"Start symbol: {start_symbol}", file=sys.stderr)
        
        # Create workflow manager and parse productions first
        workflow_manager = GrammarWorkflowManager(cfg_input)
        
        # Parse productions
        productions_result = workflow_manager.parse_productions()
        if not productions_result['success']:
            print(f"--- Production Parsing FAILED ---", file=sys.stderr)
            return jsonify({
                "success": False,
                "error": productions_result['error']
            }), 400
        
        # Set start symbol and build parse table
        result = workflow_manager.set_start_symbol(start_symbol)
        
        if result['success']:
            print("--- Parse Table Building SUCCEEDED ---", file=sys.stderr)
            print(f"States created: {result['table_info']['states_count']}", file=sys.stderr)
            print(f"Action entries: {result['table_info']['action_entries']}", file=sys.stderr)
            print(f"Goto entries: {result['table_info']['goto_entries']}", file=sys.stderr)
            
            response_data = {
                "success": True,
                "parse_table_html": result['parse_table_html'],
                "start_symbol": result['start_symbol'],
                "table_info": result['table_info']
            }
            
            # Include conflicts if any
            if result['conflicts']:
                response_data['conflicts'] = result['conflicts']
                print(f"Conflicts detected: {len(result['conflicts'])}", file=sys.stderr)
            
            return jsonify(response_data)
        else:
            print(f"--- Parse Table Building FAILED ---", file=sys.stderr)
            print(f"Error: {result['error']}", file=sys.stderr)
            
            return jsonify({
                "success": False,
                "error": result['error']
            }), 400
            
    except ImportError as e:
        print(f"--- IMPORT ERROR: Workflow manager not found: {e} ---", file=sys.stderr)
        return jsonify({"error": f"Workflow manager not available: {e}"}), 500
        
    except Exception as e:
        print(f"--- UNEXPECTED Python Error: {e} ---", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        error_message = f"Unexpected server error: {escapeHtml(str(e))}"
        return jsonify({"error": error_message}), 500

@app.route('/visualize-parser', methods=['POST'])
def visualize_parser():
    """
    Parse CFG and input string using pure Python CLR implementation.
    
    Enhanced to support workflow-based parsing with optional start symbol parameter
    and step-by-step parsing trace generation with enhanced error messages.
    """
    data = request.json
    cfg_input = data.get('cfg')
    string_input = data.get('input')
    start_symbol = data.get('start_symbol')  # Optional start symbol parameter
    
    if not cfg_input:
        return jsonify({"error": "No CFG provided"}), 400
    if not string_input:
        return jsonify({"error": "No input string provided"}), 400

    try:
        # Check if we should use workflow-based parsing or direct parsing
        if start_symbol:
            # Use workflow manager for enhanced parsing with specific start symbol
            from cfg_parser import GrammarWorkflowManager
            
            print("--- Using Workflow-based Parser Implementation ---", file=sys.stderr)
            print(f"Start symbol: {start_symbol}", file=sys.stderr)
            
            # Create workflow manager
            workflow_manager = GrammarWorkflowManager(cfg_input)
            
            # Parse productions
            productions_result = workflow_manager.parse_productions()
            if not productions_result['success']:
                print(f"--- Production Parsing FAILED ---", file=sys.stderr)
                return jsonify({
                    "error": productions_result['error'],
                    "error_type": "grammar_error"
                }), 400
            
            # Set start symbol and build parse table
            table_result = workflow_manager.set_start_symbol(start_symbol)
            if not table_result['success']:
                print(f"--- Parse Table Building FAILED ---", file=sys.stderr)
                return jsonify({
                    "error": table_result['error'],
                    "error_type": "table_generation_error"
                }), 400
            
            print("--- Parse Table Building SUCCEEDED ---", file=sys.stderr)
            
            # Parse the input string with step-by-step trace
            print(f"--- Parsing Input String: '{string_input}' ---", file=sys.stderr)
            parse_result = workflow_manager.parse_input_string(string_input)
            
            if parse_result['success']:
                print("--- Parsing SUCCEEDED ---", file=sys.stderr)
                
                # Return enhanced successful parsing results
                return jsonify({
                    "parseTreeDot": parse_result['parse_tree_dot'],
                    "parseTableHtml": table_result['parse_table_html'],
                    "parseTraceHtml": parse_result['trace_html'],
                    "traceSteps": parse_result['trace_steps'],
                    "startSymbol": start_symbol,
                    "tableInfo": table_result['table_info'],
                    "conflicts": table_result.get('conflicts', [])
                })
            else:
                print(f"--- Parsing FAILED ---", file=sys.stderr)
                print(f"Error: {parse_result['error']}", file=sys.stderr)
                
                # Determine specific error type
                error_type = "parsing_error"
                if "not parsable with the grammar" in parse_result['error'].lower():
                    error_type = "string_not_parsable"
                elif "unexpected" in parse_result['error'].lower():
                    error_type = "unexpected_token"
                elif "lexical" in parse_result['error'].lower():
                    error_type = "lexical_error"
                
                # Return enhanced parsing error with tables and trace
                return jsonify({
                    "error": parse_result['error'],
                    "error_type": error_type,
                    "error_position": parse_result.get('error_position', -1),
                    "parseTableHtml": table_result['parse_table_html'],
                    "parseTraceHtml": parse_result.get('trace_html', ''),
                    "traceSteps": parse_result.get('trace_steps', 0),
                    "startSymbol": start_symbol,
                    "tableInfo": table_result['table_info']
                }), 400
        
        else:
            # Use original CFGParserVisualizer for backward compatibility
            from cfg_parser import CFGParserVisualizer
            
            print("--- Using Original CLR Parser Implementation ---", file=sys.stderr)
            
            # Create the parser visualizer
            parser_visualizer = CFGParserVisualizer()
            
            # Process the grammar
            print(f"--- Processing Grammar ---", file=sys.stderr)
            grammar_result = parser_visualizer.process_grammar(cfg_input)
            
            if not grammar_result['success']:
                print(f"--- Grammar Processing FAILED ---", file=sys.stderr)
                print(f"Error: {grammar_result['error']}", file=sys.stderr)
                return jsonify({
                    "error": grammar_result['error'],
                    "error_type": "grammar_error"
                }), 400
            
            print("--- Grammar Processing SUCCEEDED ---", file=sys.stderr)
            
            # Parse the input string
            print(f"--- Parsing Input String: '{string_input}' ---", file=sys.stderr)
            parse_result = parser_visualizer.parse_input(string_input)
            
            if parse_result['success']:
                print("--- Parsing SUCCEEDED ---", file=sys.stderr)
                
                # Return successful parsing results
                return jsonify({
                    "parseTreeDot": parse_result['tree_dot'],
                    "parseTableHtml": grammar_result['tables_html'],
                    "parseTraceHtml": parse_result['trace_html']
                })
            else:
                print(f"--- Parsing FAILED ---", file=sys.stderr)
                print(f"Error: {parse_result['error']}", file=sys.stderr)
                
                # Determine specific error type
                error_type = "parsing_error"
                if "not parsable with the grammar" in parse_result['error'].lower():
                    error_type = "string_not_parsable"
                elif "unexpected" in parse_result['error'].lower():
                    error_type = "unexpected_token"
                elif "lexical" in parse_result['error'].lower():
                    error_type = "lexical_error"
                
                # Return parsing error with tables (grammar was valid)
                return jsonify({
                    "error": parse_result['error'],
                    "error_type": error_type,
                    "parseTableHtml": grammar_result['tables_html'],
                    "parseTraceHtml": parse_result.get('trace_html', '')
                }), 400
            
    except ImportError as e:
        print(f"--- IMPORT ERROR: CLR parser module not found: {e} ---", file=sys.stderr)
        return jsonify({
            "error": f"CLR parser implementation not available: {e}",
            "error_type": "system_error"
        }), 500
        
    except Exception as e:
        print(f"--- UNEXPECTED Python Error: {e} ---", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        error_message = f"Unexpected server error: {escapeHtml(str(e))}"
        return jsonify({
            "error": error_message,
            "error_type": "system_error"
        }), 500

# --- Main Execution ---
if __name__ == '__main__':
    print("--- Compiler Visualizer Server ---")
    print("CLR Parser Implementation - No external dependencies required")
    
    # Check for regex visualizer (optional)
    regex_available = check_executables()
    if regex_available:
        print("Regex visualizer found.")
    else:
        print("Regex visualizer not available (optional).")
    
    print(f"Running on http://127.0.0.1:5000")
    print("Open this URL in your browser.")
    print("-" * 34)
    app.run(debug=True, port=5000, use_reloader=False)