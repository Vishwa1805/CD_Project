%{
#include <stdio.h>
#include "ast.h"

extern int regexlex();
extern int regexparse();
extern FILE* regexin;
void regexerror(const char *s);

ASTNode* ast_root = NULL;
%}

%union {
    struct ASTNode* node;
}

%token <node> CHAR
%token LPAREN RPAREN PIPE STAR 

%type <node> expr term factor atom

%left PIPE                
%nonassoc CONCAT          
%left STAR                

%%

program:
    expr { ast_root = $1; YYACCEPT; }
    ;

expr:
    term                { $$ = $1; }
    | expr PIPE term    { $$ = create_op_node(NODE_UNION, $1, $3); } // Handles both '|' and '+'
    ;

term:
    factor              { $$ = $1; }
    | term factor %prec CONCAT { $$ = create_op_node(NODE_CONCAT, $1, $2); } // Implicit concatenation
    ;

factor:
    atom                { $$ = $1; }
    | factor STAR       { $$ = create_op_node(NODE_STAR, $1, NULL); } // Kleene Star (*)
    ;

atom:
    CHAR                { $$ = $1; }
    | LPAREN expr RPAREN { $$ = $2; } // Grouping
    ;

%%

void regexerror(const char *s) {
    fprintf(stderr, "Regex Parse error: %s\n", s);
}