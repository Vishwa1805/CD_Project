#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h> // For _msize on Windows with MinGW/MSVC? Or just size_t
#include "engine.h"
#include "ast.h"

// Define EPSILON for NFA transitions
#define EPSILON 0
// Define a max number of NFA states (for sets)
#define MAX_NFA_STATES 1000

/* --- Helper Data Structures (List, Set) --- */
typedef struct List { void** items; int size; int capacity; } List;
List* create_list() {
    List* list = (List*)malloc(sizeof(List));
    if (!list) { fprintf(stderr, "Malloc failed for List struct\n"); return NULL; }
    list->size = 0;
    list->capacity = 4;
    list->items = (void**)malloc(sizeof(void*) * list->capacity);
     if (!list->items) { fprintf(stderr, "Malloc failed for List items\n"); free(list); return NULL; }
    return list;
}
void list_add(List* list, void* item) {
    if (!list) return;
    if (list->size == list->capacity) {
        list->capacity *= 2;
        void** new_items = (void**)realloc(list->items, sizeof(void*) * list->capacity);
        if (!new_items) { fprintf(stderr, "Realloc failed for List items\n"); return; }
        list->items = new_items;
    }
    list->items[list->size++] = item;
}
void free_list(List* list) {
    if (list) { if (list->items) free(list->items); free(list); }
}
typedef struct Set { int states[MAX_NFA_STATES]; int size; } Set;
Set* create_set() {
    Set* set = (Set*)malloc(sizeof(Set));
     if (!set) { fprintf(stderr, "Malloc failed for Set struct\n"); return NULL; }
    set->size = 0;
    return set;
}
void set_add(Set* set, int state_id) {
    if (!set) return; if (set->size >= MAX_NFA_STATES) return;
    for (int i = 0; i < set->size; i++) { if (set->states[i] == state_id) return; }
    set->states[set->size++] = state_id;
}
int compare_ints(const void* a, const void* b) { return (*(int*)a - *(int*)b); }
int set_equals(Set* a, Set* b) {
     if (!a || !b) return 0; if (a->size != b->size) return 0;
    qsort(a->states, a->size, sizeof(int), compare_ints);
    qsort(b->states, b->size, sizeof(int), compare_ints);
    for (int i = 0; i < a->size; i++) { if (a->states[i] != b->states[i]) return 0; }
    return 1;
}
Set* set_clone(Set* set) {
     if (!set) return NULL; Set* new_set = create_set(); if (!new_set) return NULL;
    memcpy(new_set->states, set->states, set->size * sizeof(int));
    new_set->size = set->size;
    return new_set;
}
void free_set(Set* set) { free(set); }

/* --- NFA Data Structures --- */
struct NFAState;
typedef struct NFATransition { char symbol; struct NFAState* to; } NFATransition;
typedef struct NFAState { int id; int is_accept; List* transitions; } NFAState;
typedef struct NFA { NFAState* start; NFAState* accept; } NFA;
int nfa_state_id = 0;
List* all_nfa_states = NULL; // Global list
NFAState* create_nfa_state(int is_accept) {
    NFAState* state = (NFAState*)malloc(sizeof(NFAState));
    if (!state) { fprintf(stderr, "Malloc failed for NFAState\n"); return NULL; }
    state->id = nfa_state_id++; state->is_accept = is_accept; state->transitions = create_list();
    if (!state->transitions) { free(state); return NULL; }
    return state;
}
void add_nfa_transition(NFAState* from, NFAState* to, char symbol) {
     if (!from || !to || !from->transitions) return;
    NFATransition* trans = (NFATransition*)malloc(sizeof(NFATransition));
    if (!trans) { fprintf(stderr, "Malloc failed for NFATransition\n"); return; }
    trans->symbol = symbol; trans->to = to; list_add(from->transitions, trans);
}
void register_nfa_state(NFAState* state) {
    if (!state) { fprintf(stderr, "ERROR: Attempted to register NULL NFA state\n"); return; }
    if (all_nfa_states == NULL) { all_nfa_states = create_list(); if (!all_nfa_states) { fprintf(stderr, "ERROR: Failed to initialize global NFA state list\n"); return; } }
    list_add(all_nfa_states, state);
}
NFAState* get_nfa_state(int id) {
    if (all_nfa_states == NULL) return NULL;
    for (int i = 0; i < all_nfa_states->size; i++) { NFAState* s = (NFAState*)all_nfa_states->items[i]; if (s && s->id == id) return s; }
    return NULL;
}
void free_nfa_graph(NFA* nfa) {
    if (all_nfa_states) {
        for (int i = 0; i < all_nfa_states->size; i++) {
            NFAState* state = (NFAState*)all_nfa_states->items[i]; if (!state) continue;
            if (state->transitions) { for (int j = 0; j < state->transitions->size; j++) { if (state->transitions->items[j]) free(state->transitions->items[j]); } free_list(state->transitions); }
            free(state);
        }
        free_list(all_nfa_states);
    }
    if (nfa) free(nfa);
    nfa_state_id = 0; all_nfa_states = NULL;
}

/* --- DFA Data Structures --- */
struct DFAState;
typedef struct DFATransition { char symbol; struct DFAState* to; } DFATransition;
typedef struct DFAState { int id; int is_accept; Set* nfa_states; List* transitions; } DFAState;
int dfa_state_id = 0;
DFAState* create_dfa_state(Set* nfa_set) {
     if (!nfa_set) {fprintf(stderr, "ERROR: NULL nfa_set in create_dfa_state\n"); return NULL;}
    DFAState* state = (DFAState*)malloc(sizeof(DFAState)); if (!state) { fprintf(stderr, "Malloc failed for DFAState\n"); return NULL; }
    state->id = dfa_state_id++; state->nfa_states = set_clone(nfa_set); if (!state->nfa_states) { free(state); return NULL; }
    state->transitions = create_list(); if (!state->transitions) { free_set(state->nfa_states); free(state); return NULL; }
    state->is_accept = 0;
    for (int i = 0; i < nfa_set->size; i++) { NFAState* nfa_s = get_nfa_state(nfa_set->states[i]); if (nfa_s && nfa_s->is_accept) { state->is_accept = 1; break; } }
    return state;
}
void add_dfa_transition(DFAState* from, DFAState* to, char symbol) {
     if (!from || !to || !from->transitions) return;
    DFATransition* trans = (DFATransition*)malloc(sizeof(DFATransition)); if (!trans) { fprintf(stderr, "Malloc failed for DFATransition\n"); return; }
    trans->symbol = symbol; trans->to = to; list_add(from->transitions, trans);
}
void free_dfa(List* dfa_states) {
    if (dfa_states) {
        for (int i = 0; i < dfa_states->size; i++) {
            DFAState* state = (DFAState*)dfa_states->items[i]; if (!state) continue;
            if (state->transitions) { for (int j = 0; j < state->transitions->size; j++) { if (state->transitions->items[j]) free(state->transitions->items[j]); } free_list(state->transitions); }
            if (state->nfa_states) free_set(state->nfa_states); free(state);
        }
        free_list(dfa_states);
    }
    dfa_state_id = 0;
}

/* --- Thompson's Construction (ast_to_nfa) --- */
NFA* ast_to_nfa(ASTNode* ast) {
    if (!ast) { fprintf(stderr, "ERROR: ast_to_nfa called with NULL AST node.\n"); return NULL; }
    NFA* nfa_frag = (NFA*)malloc(sizeof(NFA)); if (!nfa_frag) { fprintf(stderr, "Malloc failed for NFA struct\n"); return NULL; }
    nfa_frag->start = NULL; nfa_frag->accept = NULL;
    switch (ast->type) {
        case NODE_CHAR: {
            nfa_frag->start = create_nfa_state(0); if (!nfa_frag->start) { free(nfa_frag); return NULL; }
            nfa_frag->accept = create_nfa_state(1); if (!nfa_frag->accept) { free(nfa_frag->start); free(nfa_frag); return NULL; }
            add_nfa_transition(nfa_frag->start, nfa_frag->accept, ast->data);
            register_nfa_state(nfa_frag->start); register_nfa_state(nfa_frag->accept);
            break;
        }
        case NODE_CONCAT: {
            NFA* left_nfa = ast_to_nfa(ast->left); NFA* right_nfa = ast_to_nfa(ast->right);
            if (!left_nfa || !right_nfa || !left_nfa->accept || !right_nfa->start) { fprintf(stderr, "ERROR: NULL NFA fragment in CONCAT. Left: %p, Right: %p\n", (void*)left_nfa, (void*)right_nfa); if (left_nfa) free(left_nfa); if (right_nfa) free(right_nfa); free(nfa_frag); return NULL; }
            add_nfa_transition(left_nfa->accept, right_nfa->start, EPSILON); left_nfa->accept->is_accept = 0;
            nfa_frag->start = left_nfa->start; nfa_frag->accept = right_nfa->accept;
            // Free only the container structs, not the states they point to
            free(left_nfa); free(right_nfa);
            break;
        }
        case NODE_UNION: {
            NFA* left_nfa = ast_to_nfa(ast->left); NFA* right_nfa = ast_to_nfa(ast->right);
            if (!left_nfa || !right_nfa || !left_nfa->start || !right_nfa->start || !left_nfa->accept || !right_nfa->accept) { fprintf(stderr, "ERROR: NULL NFA fragment in UNION. Left: %p, Right: %p\n", (void*)left_nfa, (void*)right_nfa); if (left_nfa) free(left_nfa); if (right_nfa) free(right_nfa); free(nfa_frag); return NULL; }
            nfa_frag->start = create_nfa_state(0); if (!nfa_frag->start) { free(left_nfa); free(right_nfa); free(nfa_frag); return NULL; }
            nfa_frag->accept = create_nfa_state(1); if (!nfa_frag->accept) { free(nfa_frag->start); free(left_nfa); free(right_nfa); free(nfa_frag); return NULL; }
            add_nfa_transition(nfa_frag->start, left_nfa->start, EPSILON); add_nfa_transition(nfa_frag->start, right_nfa->start, EPSILON);
            add_nfa_transition(left_nfa->accept, nfa_frag->accept, EPSILON); add_nfa_transition(right_nfa->accept, nfa_frag->accept, EPSILON);
            left_nfa->accept->is_accept = 0; right_nfa->accept->is_accept = 0;
            register_nfa_state(nfa_frag->start); register_nfa_state(nfa_frag->accept);
            free(left_nfa); free(right_nfa);
            break;
        }
        case NODE_STAR: {
            NFA* sub_nfa = ast_to_nfa(ast->left);
             if (!sub_nfa || !sub_nfa->start || !sub_nfa->accept) { fprintf(stderr, "ERROR: NULL NFA fragment in STAR. Sub: %p\n", (void*)sub_nfa); if (sub_nfa) free(sub_nfa); free(nfa_frag); return NULL; }
            nfa_frag->start = create_nfa_state(0); if (!nfa_frag->start) { free(sub_nfa); free(nfa_frag); return NULL; }
            nfa_frag->accept = create_nfa_state(1); if (!nfa_frag->accept) { free(nfa_frag->start); free(sub_nfa); free(nfa_frag); return NULL; }
            add_nfa_transition(nfa_frag->start, nfa_frag->accept, EPSILON); add_nfa_transition(nfa_frag->start, sub_nfa->start, EPSILON);
            add_nfa_transition(sub_nfa->accept, nfa_frag->accept, EPSILON); add_nfa_transition(sub_nfa->accept, sub_nfa->start, EPSILON);
            sub_nfa->accept->is_accept = 0;
            register_nfa_state(nfa_frag->start); register_nfa_state(nfa_frag->accept);
            free(sub_nfa);
            break;
        }
        case NODE_PLUS: {
            NFA* sub_nfa = ast_to_nfa(ast->left);
             if (!sub_nfa || !sub_nfa->start || !sub_nfa->accept) { fprintf(stderr, "ERROR: NULL NFA fragment in PLUS. Sub: %p\n", (void*)sub_nfa); if (sub_nfa) free(sub_nfa); free(nfa_frag); return NULL; }
            nfa_frag->start = create_nfa_state(0); if (!nfa_frag->start) { free(sub_nfa); free(nfa_frag); return NULL; }
            nfa_frag->accept = create_nfa_state(1); if (!nfa_frag->accept) { free(nfa_frag->start); free(sub_nfa); free(nfa_frag); return NULL; }
            add_nfa_transition(nfa_frag->start, sub_nfa->start, EPSILON); add_nfa_transition(sub_nfa->accept, nfa_frag->accept, EPSILON);
            add_nfa_transition(sub_nfa->accept, sub_nfa->start, EPSILON);
            sub_nfa->accept->is_accept = 0;
            register_nfa_state(nfa_frag->start); register_nfa_state(nfa_frag->accept);
            free(sub_nfa);
            break;
        }
        default: fprintf(stderr, "ERROR: Unknown AST node type %d in ast_to_nfa\n", ast->type); free(nfa_frag); return NULL;
    }
    return nfa_frag;
}

/* --- Subset Construction Helpers (epsilon_closure, nfa_move) --- */
void epsilon_closure_recursive(NFAState* state, Set* closure) {
    if (!state || !closure) return; set_add(closure, state->id); if (!state->transitions) return;
    for (int i = 0; i < state->transitions->size; i++) {
        NFATransition* t = (NFATransition*)state->transitions->items[i]; if (!t || !t->to) continue;
        if (t->symbol == EPSILON) {
            int already_in = 0; for(int j = 0; j < closure->size; j++) { if(closure->states[j] == t->to->id) { already_in = 1; break; } }
            if (!already_in) { epsilon_closure_recursive(t->to, closure); }
        }
    }
}
Set* epsilon_closure(Set* states) {
    Set* closure = create_set(); if (!closure) return NULL; if (!states) return closure;
    for (int i = 0; i < states->size; i++) { NFAState* state = get_nfa_state(states->states[i]); if (state) { epsilon_closure_recursive(state, closure); } }
    return closure;
}
Set* nfa_move(Set* states, char symbol) {
    Set* move_set = create_set(); if (!move_set) return NULL; if (!states) return move_set;
    for (int i = 0; i < states->size; i++) {
        NFAState* state = get_nfa_state(states->states[i]); if (!state || !state->transitions) continue;
        for (int j = 0; j < state->transitions->size; j++) { NFATransition* t = (NFATransition*)state->transitions->items[j]; if (!t || !t->to) continue; if (t->symbol == symbol) { set_add(move_set, t->to->id); } }
    }
    return move_set;
}

/* --- Main Subset Construction (nfa_to_dfa) --- */
List* nfa_to_dfa(NFA* nfa) {
    dfa_state_id = 0; List* dfa_states = create_list(); if (!dfa_states) return NULL;
    List* work_list = create_list(); if (!work_list) { free_list(dfa_states); return NULL; }
    Set* start_set = create_set(); if (!start_set) { free_list(dfa_states); free_list(work_list); return NULL; }
    if (nfa == NULL || nfa->start == NULL) { fprintf(stderr, "ERROR: NFA or NFA start state is NULL in nfa_to_dfa.\n"); free_set(start_set); free_list(dfa_states); free_list(work_list); return NULL; }
    set_add(start_set, nfa->start->id);
    Set* start_closure = epsilon_closure(start_set); free_set(start_set); if (!start_closure) { free_list(dfa_states); free_list(work_list); return NULL; }
    DFAState* dfa_start_state = create_dfa_state(start_closure); if (!dfa_start_state) { free_set(start_closure); free_list(dfa_states); free_list(work_list); return NULL; }
    list_add(dfa_states, dfa_start_state); list_add(work_list, dfa_start_state);
    while (work_list->size > 0) {
        DFAState* current_dfa_state = (DFAState*)work_list->items[--work_list->size]; if (!current_dfa_state) continue;
        for (char c = 32; c <= 126; c++) { // Consider only printable ASCII for transitions
            Set* move_set = nfa_move(current_dfa_state->nfa_states, c); if (!move_set) continue;
            if (move_set->size == 0) { free_set(move_set); continue; }
            Set* closure_set = epsilon_closure(move_set); free_set(move_set); if (!closure_set) continue;
            if (closure_set->size == 0) { free_set(closure_set); continue; }
            DFAState* target_dfa_state = NULL;
            for (int i = 0; i < dfa_states->size; i++) { DFAState* s = (DFAState*)dfa_states->items[i]; if (s && set_equals(s->nfa_states, closure_set)) { target_dfa_state = s; break; } }
            if (target_dfa_state == NULL) { target_dfa_state = create_dfa_state(closure_set); if (!target_dfa_state) { free_set(closure_set); continue; } list_add(dfa_states, target_dfa_state); list_add(work_list, target_dfa_state); }
            add_dfa_transition(current_dfa_state, target_dfa_state, c);
            free_set(closure_set);
        }
    }
    free_list(work_list); return dfa_states;
}

/* --- DOT String Generation --- */
char* escape_char_for_dot(char c) { /* ... same as before ... */
    static char buf_dot[8]; if (c == '"') sprintf(buf_dot, "\\\""); else if (c == '\\') sprintf(buf_dot, "\\\\"); else if (c == ' ') sprintf(buf_dot, "' '"); else if (c < 32 || c > 126) sprintf(buf_dot, "?"); else sprintf(buf_dot, "%c", c); return buf_dot;
}
char* dfa_to_dot_string(List* dfa_states) {
    if (dfa_states == NULL) { return strdup("digraph DFA {\n  error [label=\"DFA list is NULL\"];\n}\n"); }
    size_t current_size = 1024; char* dot_buffer = (char*)malloc(current_size); if (!dot_buffer) return strdup("digraph DFA { error [label=\"Malloc failed\"]; }");
    char* ptr = dot_buffer; char* end = dot_buffer + current_size;
    ptr += snprintf(ptr, end - ptr, "digraph DFA {\n  rankdir=LR;\n  node [shape = circle];\n  \"\" [shape=none,width=0,height=0];\n  \"\" -> S0 [label=\"start\"];\n");
    for (int i = 0; i < dfa_states->size; i++) {
        DFAState* s = (DFAState*)dfa_states->items[i]; if (!s) continue;
        if(ptr + 100 > end) { /* Resize */ size_t offset = ptr - dot_buffer; current_size *= 2; char* new_buff = (char*)realloc(dot_buffer, current_size); if(!new_buff) { free(dot_buffer); return strdup("digraph DFA { error [label=\"Realloc failed\"]; }"); } dot_buffer = new_buff; ptr = dot_buffer + offset; end = dot_buffer + current_size; }
        ptr += snprintf(ptr, end - ptr, "  S%d [shape=%s];\n", s->id, s->is_accept ? "doublecircle" : "circle");
    }
    for (int i = 0; i < dfa_states->size; i++) {
        DFAState* s = (DFAState*)dfa_states->items[i]; if (!s || !s->transitions) continue;
        List* targets = create_list(); if (!targets) continue; List* labels = create_list(); if (!labels) { free_list(targets); continue; }
        for (int j = 0; j < s->transitions->size; j++) {
            DFATransition* t = (DFATransition*)s->transitions->items[j]; if (!t || !t->to) continue;
            int target_index = -1; for(int k = 0; k < targets->size; k++) { if(targets->items[k] == t->to) { target_index = k; break; } }
            if (target_index == -1) { list_add(targets, t->to); char* label = (char*)malloc(128); if (!label) { /* handle */ break; } strcpy(label, escape_char_for_dot(t->symbol)); list_add(labels, label); }
            else { char* label = (char*)labels->items[target_index]; if (label && strlen(label) < 110) { strcat(label, ", "); strcat(label, escape_char_for_dot(t->symbol)); } }
        }
        for(int k = 0; k < targets->size; k++) {
            DFAState* target = (DFAState*)targets->items[k]; char* label = (char*)labels->items[k];
            if (!target || !label) { if(label) free(label); continue; }
            if(ptr + strlen(label) + 50 > end) { /* Resize */ size_t offset = ptr - dot_buffer; current_size *= 2; char* new_buff = (char*)realloc(dot_buffer, current_size); if(!new_buff) { free(label); free_list(targets); free_list(labels); free(dot_buffer); return strdup("digraph DFA { error [label=\"Realloc failed\"]; }"); } dot_buffer = new_buff; ptr = dot_buffer + offset; end = dot_buffer + current_size; }
            ptr += snprintf(ptr, end - ptr, "  S%d -> S%d [label=\"%s\"];\n", s->id, target->id, label); free(label);
        }
        free_list(targets); free_list(labels);
    }
    if (ptr + 5 > end) { /* Resize for closing brace */ /* ... resize logic ... */ }
    ptr += snprintf(ptr, end - ptr, "}\n");
    // Trim excess buffer (optional)
    char* final_dot = strdup(dot_buffer); free(dot_buffer);
    return final_dot;
}

/* --- JSON Generation --- */
char* escape_char_for_json(char c, char* buffer) { /* ... same as before ... */
    if (c == '"') sprintf(buffer, "\\\""); else if (c == '\\') sprintf(buffer, "\\\\"); else if (c < 32 || c == 127) sprintf(buffer, "\\u%04x", (unsigned int)c); else sprintf(buffer, "%c", c); return buffer;
}
char* dfa_to_json(List* dfa_states) {
    if (dfa_states == NULL) { return strdup("{\"error\": \"DFA list is NULL\"}"); }
    size_t current_size = 1024; char* json_buffer = (char*)malloc(current_size); if (!json_buffer) return strdup("{\"error\": \"Malloc failed\"}");
    char* ptr = json_buffer; char* end = json_buffer + current_size; int first_state = 1; char escape_buffer[8];
    ptr += snprintf(ptr, end - ptr, "{\n  \"startState\": 0,\n  \"acceptStates\": [");
    int first_accept = 1;
    for (int i = 0; i < dfa_states->size; i++) { DFAState* s = (DFAState*)dfa_states->items[i]; if (s && s->is_accept) { if(ptr + 20 > end) {/* Resize */} ptr += snprintf(ptr, end - ptr, "%s%d", first_accept ? "" : ", ", s->id); first_accept = 0; } }
    if(ptr + 50 > end) {/* Resize */} ptr += snprintf(ptr, end - ptr, "],\n  \"transitions\": {\n");
    for (int i = 0; i < dfa_states->size; i++) {
        DFAState* s = (DFAState*)dfa_states->items[i]; if (!s || !s->transitions || s->transitions->size == 0) continue; // Skip states with no transitions for cleaner JSON
        if(ptr + 50 > end) {/* Resize */} ptr += snprintf(ptr, end - ptr, "%s    \"%d\": {", first_state ? "" : ",\n", s->id); int first_trans = 1;
        // Collect transitions for the current state
        for (int j = 0; j < s->transitions->size; j++) {
            DFATransition* t = (DFATransition*)s->transitions->items[j]; if (!t || !t->to) continue;
             if(ptr + 30 > end) {/* Resize */}
             ptr += snprintf(ptr, end - ptr, "%s      \"%s\": %d", first_trans ? "\n" : ",\n", escape_char_for_json(t->symbol, escape_buffer), t->to->id);
             first_trans = 0;
        }
        if(ptr + 10 > end) {/* Resize */} ptr += snprintf(ptr, end - ptr, "\n    }"); first_state = 0;
    }
    if(ptr + 10 > end) {/* Resize */} ptr += snprintf(ptr, end - ptr, "\n  }\n}\n");
    char* final_json = strdup(json_buffer); free(json_buffer); return final_json;
}

/* --- JSON Escaping for Embedding --- */
char* escape_string_for_json_embedding(const char* str) {
    if (!str) return strdup("");
    size_t len = strlen(str);
    size_t new_len = len;
    // Count necessary escapes
    for (size_t i = 0; i < len; ++i) { if (str[i] == '"' || str[i] == '\\' || str[i] == '\n' || str[i] == '\r') new_len++; }
    char* escaped = (char*)malloc(new_len + 1); if (!escaped) return strdup("");
    char* dst = escaped; const char* src = str;
    while (*src) {
        if (*src == '"') { *dst++ = '\\'; *dst++ = '"'; }
        else if (*src == '\\') { *dst++ = '\\'; *dst++ = '\\'; }
        else if (*src == '\n') { *dst++ = '\\'; *dst++ = 'n'; }
        else if (*src == '\r') { /* skip */ }
        else { *dst++ = *src; }
        src++;
    }
    *dst = '\0'; return escaped;
}

/* --- Main Entry Point --- */
void process_regex_to_dfa_dot(ASTNode* root) {
    char* final_output_json = NULL;
    char* error_json = NULL;
    NFA* nfa = NULL;
    List* dfa = NULL;
    char* dot_string = NULL;
    char* json_table_string = NULL;
    char* escaped_dot = NULL;

    if (!root) { error_json = strdup("{\"error\": \"AST root was NULL\"}"); goto cleanup; }

    nfa = ast_to_nfa(root);
    if (!nfa) { error_json = strdup("{\"error\": \"NFA generation failed (ast_to_nfa returned NULL)\"}"); goto cleanup; }

    dfa = nfa_to_dfa(nfa);
    if (!dfa) { error_json = strdup("{\"error\": \"DFA generation failed (nfa_to_dfa returned NULL)\"}"); goto cleanup; }

    dot_string = dfa_to_dot_string(dfa);
    if (!dot_string) { error_json = strdup("{\"error\": \"DOT string generation failed\"}"); goto cleanup; }

    json_table_string = dfa_to_json(dfa);
    if (!json_table_string) { error_json = strdup("{\"error\": \"JSON table generation failed\"}"); goto cleanup; }

    escaped_dot = escape_string_for_json_embedding(dot_string);
    if (!escaped_dot) { error_json = strdup("{\"error\": \"Failed to escape DOT string\"}"); goto cleanup; }

    // Allocate buffer for the final combined JSON
    size_t final_size = strlen(escaped_dot) + strlen(json_table_string) + 50; // Estimate
    final_output_json = (char*)malloc(final_size);
    if (!final_output_json) { error_json = strdup("{\"error\": \"Malloc failed for final JSON output\"}"); goto cleanup; }

    // Construct the final JSON
    snprintf(final_output_json, final_size, "{\n  \"dot\": \"%s\",\n  \"tableData\": %s\n}",
             escaped_dot, json_table_string);

    // Print the final result
    printf("%s\n", final_output_json);

cleanup:
    // Print error JSON if one occurred
    if (error_json) {
        fprintf(stderr, "%s\n", error_json); // Print error to stderr as well
        printf("%s\n", error_json);          // Print error JSON to stdout
        free(error_json);
    }

    // Free all allocated memory
    if (final_output_json) free(final_output_json);
    if (escaped_dot) free(escaped_dot);
    if (json_table_string) free(json_table_string);
    if (dot_string) free(dot_string);
    if (nfa) free_nfa_graph(nfa); // free_nfa_graph handles NULL check for nfa itself
    if (dfa) free_dfa(dfa);       // free_dfa handles NULL check
}