# Principles of Reactive Programming

## Week 1
### What is reactive programming?
- Driven by changing requirements
	- more server nodes
	- lower response time
	- zero down time
	- larger data volume
- New architecture
	- react to events (event-driven)
	- react to load (scalable)
	- react to failure (resilient)
	- react to users (responsive)
- Event-driven
	- traditionally, multiple threads which communicate with shared synchronized states
		- strong coupling, hard to compose
	- event driven systems are composed of loosely coupled event handlers
		- events can be handled asynchronously, `without blocking`
- Scalable
	- able to expand according to its usage
		- scale up: make use of parallelism of multi-core system
		- scale out: make use of multiple server nodes
	- important for scalability: minimized shared states
	- important for scale out: location transparency, resiliency
- Resilient
	- recover quickly from failures
	- needs to be part of the design
		- loose coupling
		- strong encapsulation of state
		- pervasive supervisor hierarchies
- Responsive
	- provide rich, real-time intereaction with users even under load and failure
- Callbacks
	- needs shared mutable states
	- cannot be composed
	- a large number of callbacks is hard to track
- Solution: use fundamental constructions from funcational programming to get composable event abstractions
	- events are first class
	- event handlers are first class
	- events represented as messages
	- complex handlers composed of primitive ones

### Functions and Pattern Matching
- functions are objects

```scala
trait Function1[-A, +R] {
	def apply(A): R
}
```
- The matching block ...

```scala
{ case (key, value) => key + ": " + value }
```
- ... is expanded to

```scala
type JBinding = (String, JSON)
new Function1[JBinding, String] {
	def apply(x: JBinding) = x match {
		case (key, value) => key + ": " + value
	}
}
```
- function classes can be subclassed
- partial functions

```scala
trait PartialFunction[-A, +R] extends Function1[-A, +R] {
	def apply(x: A): R
	def isDefinedAt(x: A): Boolean
}
```
- if expected type is a PartialFunction, scala will expand as follows

```scala
// { case "ping" => "pong" }
new PartialFunction[String, String] {
	def apply(x: String) = x match {
		case "ping" => "pong"
	}

	def isDefinedAt(x: String) = x match {
		case "ping" => true
		case _ => false
	}
}
```

### Monads
- a parametric type M[T] with `flatMap` and `unit`

```scala
trait M[T] {
	def flatMap[U](f: T => M[U]): M[U]
}

def unit[T](x: T): M[T]
```

- In literature, `flatMap` is commonly called `bind`
- List, Set, Some Random Generators are monads

```scala
unit(x) = List(x)
unit(x) = Set(x)
unit(x) = Some(x)
unit(x) = single(x)

// flatMap is an operation on each of these types
// unit in scala is different for each monad
```
- `map` can be defined in terms of `flatMap` and `bind`

```scala
m map f == m flatMap (x => unit(f(x)))
        == m flatMap (f andThen unit)
```
- three laws for monad
	
```scala
// associativity
m flatMap f flatMap g = m flatMap (x => f(x) flatMap g)

// left unit
unit(x) flatMap f == f(x)

// right unit
m flatMap unit == m
```
- associativity says one can inline nested for expressions

```scala
for(y <- for(x <- m; y <- f(x)) yield y
    z <- g(y)) yield z

== for(x <- m
       y <- f(x)
       z <- g(y)) yield z
```
- right unit rule says

```scala
for(x <- m) yield x
== m
```
- many other types defining `flatMap` are monads
- monads give useful guideline for designing library API

## Week 2
### Functions and State
- Use substitution to rewrite functions
- Rewriting can be done anywhere in the term, and all rewritings which terminate lead to the same solution
- Stateful object
	- World as a set of objects, some of which has states that change over time
	- An object has a state if its behavior is influenced by its history
	- Every form of mutable state is constructed from variables
	- In Scala, use `var` instead of `val`

```Scala
var x: String = "abc"
x = "def"
```
### Identity and Change
- Without assignments, `var x = E; var y = E` are the same, where E is arbitrary expression. It can be substituded with `var x = E; var y = x`
- This is called referential transparency
- With assignments, the two are different
- "being same" precisely means operational equivalence
	- Suppose we have two definitions x and y, they are operational equivalence if no possible tests can distinguish them
- Sustitution mode cannot be used

### Loops
- While loop
	- The condition and command must be passed by name so that they are reevaluted in each iteration
	- It is a tail recursion call
```Scala
def WHILE(condition: => Boolean)(command: => Unit): Unit =
	if(condition) {
		command
		WHILE(condition)(command)
	}
	else ()

def REPEAT(command: => Unit)(condition: => Unit): Unit = {
	command
	if(condition) ()
	else REPEAT(command)(condition)
}
```

- For loop in classical Java program cannot be modeled in Scala
	- The arguments of for contain declaration of the variable *i* which is visible in other arguments and in the body]
- Scala has a form of for loop

```Scala
for(i <- until 3) { System.out.println(i + " ") }
```
- For loop translation is similar to for-expression, but uses `foreach` combinator

```Scala
for(i <- until 3; j <- "abc") println(i + " " + j)

// Translates to
(1 until 3) foreach (i => "abc" foreach (j => println(i + " " + j)))
