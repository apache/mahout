/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
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
