Compilers
Week One
- Two implementations for programs
  - Interpreters (online): Takes data and program and gives the output
  - Compilers (offline): Takes program and produces an executable, which given data will produces the output
- Fortan I is the first compiler. Modern compilers perserve its outline
  - Lexical analysis
  - Parsing
  - Semantica analysis
  - Optimization
  - Code generation
- Lexical analysis
  - Divide program texts into words or tokens
- Parsing
  - To a diagramming structure (tree)
- Semantica analysis
  - Compilers check for inconsistency
  - Languages have strict rules to avoid ambiguities
- Optimization
  - Run faster
  - Less memory
- Code generation
  - Translate high level language to assembly code (usually)

Week Two
Lexical analysis
- Classify program string into token classes according to rule
- String -> lexical analysis -> <token class, substring>
  - Scan from left to right. Lookahead might be needed to decide boundaries of tokens.
  - We want to limit the amount the lookahead to a constant to simplify lexical analysis.
- Regular language is used to decide what token class a string belongs to
  - Single character. eg. 'c' = {'c'}
  - Epsilon. eg. e = {''}
  - Union: A and B = {a in A} or {a in B}
  - Concatenation: AB = {ab where a in A and b in B}
  - Iteration: A* = e + A + AA + AAA + ...
  - The smallest set of expressions including the 5 cases above over a given alphabet is the regular experssions over the alphabet
- Formal language is defined by a alphabet and a meaning function that maps syntax to semantics
  - Meaning function separates syntax and semantics
  - Meaning function allows us to consider notation as a separate issue
  - Meaning function is many to one. It means 1) Code optimization can be done to substitue a slower expression with a faster one while keeping the semantics 2) The same syntax cannot map to different semantics (ambiguous)
