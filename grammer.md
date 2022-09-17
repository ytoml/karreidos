# Grammer
```
program := top_level_expr
        | ext
        | func

top_level_expr := stmt

expr := primary bin_rhs

stmt := (decl | expr | block)? ';'

decl := "let" var "=" expr

var := ("mut")? ident

primary := ident_expr
        | num_expr
        | paren_expr
        | contional
        | for_expr

block := '{' stmt* '}'

conditional := "if" expr block
               ("else" (conditional | block)*

for_expr := "for" var "<-" expr ".." expr ',' expr block

bin_rhs := (bin_op primary)*

bin_op := /* refer to parser::ast::BinOp */

num_expr := number /* only float for now */

paren_expr := '(' expr ')'

ident_expr := ident
            | ident '(' (expr ',')* (expr)? (',')?  ')'

proto := ident '(' (var ',')* ident ')'

func := "fn" proto block

ext := "extern" "fn" proto ";"
```
