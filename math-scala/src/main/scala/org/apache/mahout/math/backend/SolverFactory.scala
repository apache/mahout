package org.apache.mahout.math.backend

import scala.collection.{Iterable, Map}
import scala.reflect.{ClassTag, classTag}

/**
  * == Overview ==
  *
  * Solver factory is an essence a collection of lazily initialized strategy singletons solving some
  * (any) problem in context of the Mahout project.
  *
  * We intend to use it _mainly_ for problems that are super-linear problems, and often involve more
  * than one argument (operand).
  *
  * The main method to probe for an available solver is [[RootSolverFactory.getSolver]].
  */
trait SolverFactory {
  /**
    * We take an implicit context binding, the classTag, of the trait of the solver desired.
    *
    * == Note to callers ==
    *
    * Due to Scala semantics, it is usually not enough to request a solver via merely {{{
    *   val s:SolverType = backend.getSolver
    * }}} but instead requires an explicit solver tag, i.e.: {{{
    *   val s = backend.getSolver[SolverType]
    * }}}
    *
    *
    */
  def getSolver[S: ClassTag]: Option[S] = {
    solverMap.get(classTag[S]).flatMap {
      _ match {
        case s: S ⇒ Some(s)
        case _ ⇒ None
      }
    }
  }

  lazy val availableSolverTags: Iterable[ClassTag[_]] = solverMap.keySet



  protected[backend] val solverMap: Map[ClassTag[_], Any]

  protected[backend] def validateMap(): Unit = {

    for ((tag, instance) ← solverMap) {
      require(tag.runtimeClass.isAssignableFrom(instance.getClass),
        s"Solver implementation class `${instance.getClass.getName}` is not a subclass of solver trait `${tag}`.")

    }
  }

}
