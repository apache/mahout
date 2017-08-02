

/**
  * This is a small ADT that we use to avoid building too many
  * intermediate vectors.
  *
  * It allows us to concatenate a whole bunch of vectors or single
  * elements cheaply, and then iterate over them later.
  */
package org.apache.mahout.math.algorithms.clustering.RTree

sealed trait ADT[A] extends Iterable[A] {

  def iterator: Iterator[A]

  override def isEmpty: Boolean = false

  def plus(that: ADT[A]): ADT[A] =
    if (that.isEmpty) this else ADT.Concat(this, that)

  override def hashCode(): Int =
    iterator.foldLeft(0x0a704453)((x, y) => x + (y.## * 0xbb012349 + 0x337711af))

  override def equals(that: Any): Boolean =
    that match {
      case that: ADT[_] =>
        val it1 = this.iterator
        val it2 = that.iterator
        while (it1.hasNext && it2.hasNext) {
          if (it1.next != it2.next) return false //scalastyle:off
        }
        it1.hasNext == it2.hasNext
      case _ =>
        false
    }
}

object ADT {
  def empty[A]: ADT[A] = Wrapped(Vector.empty)
  def apply[A](a: A): ADT[A] = Single(a)
  def wrap[A](as: Vector[A]): ADT[A] = Wrapped(as)

  case class Single[A](a: A) extends ADT[A] {
    def iterator: Iterator[A] = Iterator(a)
  }

  case class Wrapped[A](as: Vector[A]) extends ADT[A] {
    override def isEmpty: Boolean = as.isEmpty
    def iterator: Iterator[A] = as.iterator
    override def plus(that: ADT[A]): ADT[A] =
      if (this.isEmpty) that else if (that.isEmpty) this else ADT.Concat(this, that)
  }

  case class Concat[A](x: ADT[A], y: ADT[A]) extends ADT[A] {
    def iterator: Iterator[A] = x.iterator ++ y.iterator
  }
}

