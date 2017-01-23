package org.apache.mahout.math.backend.jvm

import org.apache.mahout.math._
import scalabindings._
import RLikeOps._
import org.apache.mahout.math.backend.Backend
import org.apache.mahout.math.scalabindings.MMul

import scala.collection.Map
import scala.reflect._

object JvmBackend extends Backend {

  import org.apache.mahout.math.backend.incore._

  /**
    * If backend has loaded (lazily) ok and verified its availability/functionality,
    * this must return `true`.
    *
    * @return `true`
    */
  override def isAvailable: Boolean = true

  // TODO: In a future release, Refactor MMul optimizations into this object
  override protected[backend] val solverMap: Map[ClassTag[_], Any] = Map(
    classTag[MMulSolver] → MMul
    //    classTag[MMulDenseSolver] → MMul,
    //    classTag[MMulSparseSolver] → MMul,
    //    classTag[AtASolver] → new AtASolver {
    //      override def apply(a: Matrix, r: Option[Matrix]): Matrix = MMul(a.t, a, r)
    //    }// ,
    //    classTag[AtADenseSolver] → { (a: Matrix, r: Option[Matrix]) ⇒ MMul(a.t, a, r) },
    //    classTag[AtASparseSolver] → { (a: Matrix, r: Option[Matrix]) ⇒ MMul(a.t, a, r) },
    //    classTag[AAtSolver] → { (a: Matrix, r: Option[Matrix]) ⇒ MMul(a, a.t, r) },
    //    classTag[AAtDenseSolver] → { (a: Matrix, r: Option[Matrix]) ⇒ MMul(a, a.t, r) },
    //    classTag[AAtSparseSolver] → { (a: Matrix, r: Option[Matrix]) ⇒ MMul(a, a.t, r) }
  )
  validateMap()

  private val mmulSolver = new MMulSolver with MMulDenseSolver with MMulSparseSolver {
    override def apply(a: Matrix, b: Matrix, r: Option[Matrix]): Matrix = MMul(a, b, r)
  }

  private val ataSolver = new AtASolver with AtADenseSolver with AtASparseSolver {
    override def apply(a: Matrix, r: Option[Matrix]): Matrix = MMul(a.t, a, r)
  }

  private val aatSolver = new AAtSolver {
    override def apply(a: Matrix, r: Option[Matrix]): Matrix = MMul(a, a.t, r)
  }
}
