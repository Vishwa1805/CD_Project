# Compiler
CC = gcc
# Flags
CFLAGS = -Wall -g

# Flex and Bison
LEX = flex
YACC = bison
YACC_FLAGS = -d -v # -v is crucial, it creates the .output file

# --- Part 1: Regex Visualizer ---
REGEX_DIR = part1_regex
REGEX_EXE = regex_visualizer.exe # Added .exe
REGEX_LEX = $(REGEX_DIR)/regex_lexer
REGEX_YACC = $(REGEX_DIR)/regex_parser
REGEX_C_FILES = $(REGEX_DIR)/main_regex.c $(REGEX_DIR)/engine.c
REGEX_GEN_C = $(REGEX_YACC).tab.c $(REGEX_LEX).yy.c
REGEX_OBJ = $(REGEX_C_FILES:.c=.o) $(REGEX_GEN_C:.c=.o)

$(REGEX_EXE): $(REGEX_OBJ)
	$(CC) $(CFLAGS) -o $(REGEX_EXE) $(REGEX_OBJ)

# --- Part 2: Calc Visualizer ---
CALC_DIR = part2_calc
CALC_EXE = calc_visualizer.exe # Added .exe
CALC_LEX = $(CALC_DIR)/calc_lexer
CALC_YACC = $(CALC_DIR)/calc_parser
CALC_C_FILES = $(CALC_DIR)/main_calc.c
CALC_GEN_C = $(CALC_YACC).tab.c $(CALC_LEX).yy.c
CALC_OBJ = $(CALC_C_FILES:.c=.o) $(CALC_GEN_C:.c=.o)

$(CALC_EXE): $(CALC_OBJ)
	$(CC) $(CFLAGS) -o $(CALC_EXE) $(CALC_OBJ)

# --- Phony Targets ---
all: $(REGEX_EXE) $(CALC_EXE)

# Changed 'rm -f' to '-del' and used Windows paths '\'
clean:
	-del $(REGEX_EXE) $(CALC_EXE)
	-del $(REGEX_DIR)\*.o $(CALC_DIR)\*.o
	-del $(REGEX_DIR)\regex_parser.tab.c $(REGEX_DIR)\regex_parser.tab.h $(REGEX_DIR)\regex_lexer.yy.c
	-del $(CALC_DIR)\calc_parser.tab.c $(CALC_DIR)\calc_parser.tab.h $(CALC_DIR)\calc_lexer.yy.c
	-del $(CALC_DIR)\calc_parser.output

# --- Build Rules ---

# Rule for Yacc (Part 1)
$(REGEX_YACC).tab.c $(REGEX_YACC).tab.h: $(REGEX_YACC).y $(REGEX_DIR)/ast.h
	$(YACC) $(YACC_FLAGS) -p regex -o $(REGEX_YACC).tab.c $(REGEX_YACC).y

# Rule for Lex (Part 1)
$(REGEX_LEX).yy.c: $(REGEX_LEX).l $(REGEX_YACC).tab.h
	$(LEX) -P regex -o $(REGEX_LEX).yy.c $(REGEX_LEX).l

# Rule for Yacc (Part 2)
$(CALC_YACC).tab.c $(CALC_YACC).tab.h: $(CALC_YACC).y $(CALC_DIR)/tree.h
	$(YACC) $(YACC_FLAGS) -p calc -o $(CALC_YACC).tab.c $(CALC_YACC).y

# Rule for Lex (Part 2)
$(CALC_LEX).yy.c: $(CALC_LEX).l $(CALC_YACC).tab.h
	$(LEX) -P calc -o $(CALC_LEX).yy.c $(CALC_LEX).l


# Modern pattern rule for building .o files from .c files in the regex dir
$(REGEX_DIR)/%.o: $(REGEX_DIR)/%.c
	$(CC) $(CFLAGS) -I$(REGEX_DIR) -c $< -o $@

# Modern pattern rule for building .o files from .c files in the calc dir
$(CALC_DIR)/%.o: $(CALC_DIR)/%.c
	$(CC) $(CFLAGS) -I$(CALC_DIR) -c $< -o $@