# Functional Programming Principles in Scala

## Week 5: Lists
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
- pair and tuple: (nesting) pattern matching

```scala
	def merge(xs: List[Int], ys: List[Int]): List[Int] = (xs, ys) match {
	  case (Nil, _) => ys
	  case (_, Nil) => xs
	  case (x::xss, y::yss) => if (x < y) x :: merge(xss, ys) else y :: merge(xs, yss)
	}
```
- Parameterization: `scala.math.Ordering[T]`

```scala
def msort[T](xs: List[T], ys: List[T])(ord: Ordering[T]): List[T] =
	...
msort(nums)((x, y) => x < y)
```
-	Implicit parameters: avoid passing certain parameters
	- Compiler figures out the right implicit definiton based on demanded type
	- The implicit definition must be marked as implicit, has a compatible type T and visible or defined in a companion object associated with type T

```scala
def msort[T](xs: List[T], ys: List[T])(implicit ord: Ordering[T]): List[T] =
	...
msort(nums)
```
- Higher-order Functions
	- `map`: transform elements in a collection
	- `filter`: selecting all elements satisfying a condition
	- `filter`, `filterNot`, `takeWhile`, `dropBy`, `span`
	- `reduceLeft`: inserts binary operator between adjacent elements in list
	- (x, y) => x * y is equivalent to (_ * _)
	- `foldLeft`: takes an additional accumulator, which is returned when called on an empty list
	- `reduceRight`, `foldRight`
- Reasoning about Concat
	- Refrential Transparency: a term is equivalent to what it reduces
	- Prove `concat` is correct: Structure Induction
		- Show P(Nil) is correct
		- Show P(x :: xs) holds if P(xs) holds
	- Fold and unfold technique
