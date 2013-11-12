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
