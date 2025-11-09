%{
#include <stdio.h>
#include "regex_parser.tab.h"
#include "ast.h"

extern YYSTYPE regexlval;
%}

%option noyywrap
%option prefix="regex"

%%

"*"         { return STAR; }
"+"         { return PIPE; } 
"|"         { return PIPE; } 
"("         { return LPAREN; }
")"         { return RPAREN; }
[a-zA-Z0-9] {
              regexlval.node = create_char_node(yytext[0]);
              return CHAR;
            }
\n          { return 0; /* End of input */ }
.           { /* Ignore invalid chars */ }

%%