#ifndef AST_H
#define AST_H

// Node types for the Regex AST
typedef enum {
    NODE_CHAR,
    NODE_STAR,   // * (zero or more)
    NODE_PLUS,   // + (one or more)
    NODE_UNION,  // | (or)
    NODE_CONCAT  // ab (and)
} NodeType;

// AST Node
typedef struct ASTNode {
    NodeType type;
    char data;          // For NODE_CHAR
    struct ASTNode *left;
    struct ASTNode *right;
} ASTNode;

// Function prototypes for AST creation
ASTNode* create_char_node(char data);
ASTNode* create_op_node(NodeType type, ASTNode* left, ASTNode* right);
void print_ast(ASTNode* node); // For debugging
void free_ast(ASTNode* node);

// This is the main entry point you will call from main()
// It will call your engine functions
void process_regex_to_dfa_dot(ASTNode* root);

#endif