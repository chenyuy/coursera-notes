# Compilers

## Introduction
#### Compiler and Interpreter
- Compiler: takes a program and output an executable, which can be run separately with data and outputs result
  - offline: the executable can be run with different data without being compiled again
- Interpreter: takes a program and data and outputs the result
  - online: running the whole program on the fly

## Structure of a compiler
#### Lexical analysis
  - Divide program text into 'words' or 'tokens'

#### Parsing
  - UnderstanD program structure

#### Semantic analysis
  - Understand meaning, compiler can do limited analysis, usually catch inconsistency

#### Optimization
  - Automatically modify programs so that they run faster and use less memory, etc.

#### Code generation
  - Translate to other language (usually assembly)
