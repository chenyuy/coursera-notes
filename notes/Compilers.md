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

## Lexical analysis
#### Goal
- Recognize substrings corresponding to *tokens*
  - Substrings are called the lexemes
- Identify the *token class* of each lexeme
  - Output a series of <token class, lexeme>, each one called *token*
- Doing left-right scan of the input => lookahead sometimes required

#### Token class
- Identifier
- Integer
- Keyword
- Whitespacee

#### Regular language
- Used determine what token class a set of strings is in
- *Regular experssions* specify regular language

#### Regular experssions
- Two base cases
  - Single character
  - Epislon - empty string
- Three compound experssions
  - Union - A + B = {a | a in A} union {b |b in B}
  - Concatenation - AB = {ab | a in A and b in B}
  - Iteration
    - A* = union(i >= 0)(A^i)
    - A concatenated with itself i times
    - Epsilon when i = 0

#### Formal language
- Definition: Let *Sigma* be a set of characters(an alphabet). A language over *Sigma* is a set of strings of characters drawn from *Sigma*
- Meaning function L maps syntax to semantics
  - For example, in regular experssions, L: Exp -> Sets of Strings
    - L(A + B) = L(A) union L(B)
    - L(AB) = {ab | a in L(A) and b in L(B)}
  - Makes clear what is syntax, what is semantics
  - Allows us to consider notation as a separate issue
  - Expressions and meanings are not 1 to 1
    - Multiple expressions have the same meaning
    - Basis for compiler optimization

#### Lexical specification
- Use regular language to specify a language
- An example for PASCAL
  - digit = '0' + '1' +'2' + '3' + '4' + '5' + '6' + '7' + '8' + '9'
  - digits = digit+ (same as digitdigit*)
  - opt_fraction = ('.'digits)? (same as ('.digits') +epsilon)
  - opt_exponent = ('E'('+' +'-' + epsilon)digits) + epsilon
  - num = digits opt\_fraction opt\_exponent
- Steps:
  - Write rexp for lexemes of each token class
  - Construct R which matches all lexemes for all tokens
  - Let input be x1x2...xn, for 1 <= i <= n check if x1...xi in L(R)
  - If success we know x1...xi in L(Rj) for some j
  - Remove x1...xi and go back to step 3
- Issues
  - Do maximal match
  - Match regex with highest priority first
  - Have an error regx which matches everything not in R

#### Finite automata
- Implementation of regular language
- DFA has no epsilon move; it accepts if ended in accept state
- NFA has epsilon move: it accepts if any of the possible moves ends in accept state
- DFA is faster to execute, but an equivalent NFA is smaller
