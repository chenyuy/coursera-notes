# Functional Programming Principles in Scala

## Week 5: Lists
### List Functions
- `length`, `last`, `init`, `take n`, `drop n`, `xs(n)`, `++` (concatenation), `reverse`, `updated(n, x)`, `indexOf x`, `contains x`
- Takes time propotional to the length of list
- Quiz

```scala
	def flatten(xs: List[Any]): List[Any] = xs match {
	  case List() => List()
	  case y::ys => y match {
	    case z:List[Any] => flatten(z) ++ flatten(ys)
	    case _ => y :: flatten(ys)
	  }
	}
```
### Pair and Tuple
- (nesting) pattern matching

```scala
	def merge(xs: List[Int], ys: List[Int]): List[Int] = (xs, ys) match {
	  case (Nil, _) => ys
	  case (_, Nil) => xs
	  case (x::xss, y::yss) => if (x < y) x :: merge(xss, ys) else y :: merge(xs, yss)
	}
```
### Parameterization

```scala
def msort[T](xs: List[T], ys: List[T])((T, T) => Boolean): List[T] =
	...
msort(nums)((x, y) => x < y)
```
- `scala.math.Ordering[T]`

```scala
def msort[T](xs: List[T], ys: List[T])(ord: Ordering[T]): List[T] =
	...
msort(nums)((x, y) => x < y)
```
### Implicit parameters
- Avoid passing certain parameters
- Compiler figures out the right implicit definiton based on demanded type
- The implicit definition must be marked as implicit, has a compatible type T and visible or defined in a companion object associated with type T

```scala
def msort[T](xs: List[T], ys: List[T])(implicit ord: Ordering[T]): List[T] =
	...
msort(nums)
```
### Higher-order Functions
- Abstract out common operation patterns on collections
- `map`: transform elements in a collection
- `filter`: selecting all elements satisfying a condition
- `filter`, `filterNot`, `takeWhile`, `dropBy`, `span` (combine `takeWhile` and `dropBy`)
- `reduceLeft`: inserts binary operator between adjacent elements in list
- (x, y) => x * y is equivalent to (_ * _)
- `foldLeft`: takes an additional accumulator, which is returned when called on an empty list
- `reduceRight`, `foldRight`

### Reasoning about Concat
- Refrential Transparency: a term is equivalent to what it reduces
- Prove `concat` is correct: Structure Induction
	- Show P(Nil) is correct
	- Show P(x :: xs) holds if P(xs) holds
- Fold and unfold technique

## Week 6
### Other Collections Beyond List
- `vector`: more blanced access pattern
	- shallow trees
	- Array of up to 32 elements; if more than 32 elements, each element becomes a pointer to an array of up to 32 elements, and so on
	- Number of access required to an index is log32(N)
	- Good for bulk operations (32 fits to the size of cache line, all elements in the same cache line so access is pretty fast; for List it is not guaranteed)
	- Immutable
	- `+:` and `:+` require creating a new object for each a level of the tree
	- Bad for recursive access pattern

```scala
	val nums = Vector(1, 2, 3)
	x +: xs
	xs :+ x
```
- `Seq`: base class for all sequences (like `List` and `Vector`)
	- Its base class is `Iterable`
	- Array and String can be implicitly converted to sequence, but are not subclasses of `Seq`
	- Some functions: `exists`, `forall`, `zip`, `unzip`, `flatMap`, `sum`, `product`, `max`, `min`
- `Range`: a sequence of evenly spaced integers
	- Stored as an object with three fields: lower(upper) bound, step

```scala
	var r: Range = Range 1 until 5 // 1, 2, 3, 4
	var s: Range = Range 1 to 5 // 1, 2 , 3, 4, 5
	1 to 10 by 3 // 1, 3, 7, 10
	6 to 1 by -2 // 6, 4, 2
```
### For-expression
- For-expression: make understanding high order function expressions easier
	- Builds a list of results of all iterations
	- Acts like a query language
	- Translated to `map`, `flatmap` and a lazy variant `filter` (`withFilter`); if user defined type has these methods defined, it can use for-expression too

```scala
// for (s) yield e where s is generators and filters
// for {s} yield e so s can be in multiple lines without writing semicolons
for (p <- persons if p.age > 20) yield p.name
// equivalent to persons filter (p => p.age > 20) map (p => p.name)

// A sample query
{
	for {
		b1 <- books
		b2 <- books
		if b1.title < b2.title
		a1 <- b1.authors
		a2 <- b2.authors
		if a1 == a2
	} yield a1
}.distinct

for(x <- e1) yield e2
// translated to
e1.map(x => e2)

for(x <- e1 if f; s) yield e2
// translated to
for(x <- e1.withFilter(x => f); s) yield e2
// filter is not immediately applied
// but all subsequent map or flatMap will use the filter f

for(x <- e1; y <- e2; s) yield e3
// translated to
e1.flatMap(x => for(y <- e2; s) yield e3)
```

### Sets
- Subclass of `Iterable`
- Most operations of `Seq` also available for `Set`
- Unordered
- No duplicates
- Fundamental operation is contains: `s contains 5`

```scala
// N-Queens problem

def isSafe(col: Int, queens: List[Int]): Boolean = {
	val row = queens.length
	val queensWithRows = (row - 1 to 0 by -1) zip queens
	queensWithRows forall {
		case (r, c) => col != c && math.abs(col - c) != row - r
	}
}

def queens(n: Int): Set[List[Int]] = {
	def placeQueens(k: Int): Set[List[Int]] =
		if (k == 0) Set(List())
		else
			for {
				queens <- placeQueens(k - 1)
				col <- 0 until n
				if isSafe(col, queens)
			} yield col :: queens
	placeQueens(n)
}
```

### Map
- Type Map[Key, Value]
- Also extends type Key => Value
	- m("key") => "value"
	- val m = Map('a' -> "1", 'b' -> "2", 'c' -> "3");  "abc" map m => "123"
- m get "key" => None if "key" not exists or Some "value"
- `sorted`: natural ordering; `sortWith`: pass in a cmp function
- `groupBy`: return a map of collections according to a discriminator function f
- `withDefaultValue`: m withDefaultValue "unknown" returns "unknown" if the key is not found in the map

```scala
// * means a varible number of parameters
// The parameters are represented as a List
def fun(bindings: (Int, Double)*) = ???
```

### Option
- None | Some value
- Use pattern matching for it