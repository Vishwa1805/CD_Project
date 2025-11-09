#ifndef ENGINE_H
#define ENGINE_H

#include "ast.h"

/*
 * ==========================================================
 * DATA STRUCTURE DECLARATIONS
 * These are needed so the compiler understands what "List*" and "NFA*" are.
 * ==========================================================
 */

// Simple List (dynamic array)
typedef struct List List;

// NFA (Thompson's Construction) Data Structures
typedef struct NFA NFA;


/*
 * ==========================================================
 * FUNCTION PROTOTYPES (Updated)
 * ==========================================================
 */

// Function prototypes
NFA* ast_to_nfa(ASTNode* ast);       // Thompson's Construction
List* nfa_to_dfa(NFA* nfa);   // Subset Construction
// void  dfa_to_dot(List* dfa_states); // Changed: Now handled by process function
void process_regex_to_dfa_dot(ASTNode* root); // Main process function


#endif