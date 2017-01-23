package org.apache.mahout.math.backend

import org.apache.mahout.math.scalabindings.{MMBinaryFunc, MMUnaryFunc}

package object incore {

  trait MMulSolver extends MMBinaryFunc
  trait MMulDenseSolver extends MMulSolver
  trait MMulSparseSolver extends MMulSolver
  trait AAtSolver extends MMUnaryFunc
  trait AAtDenseSolver extends AAtSolver
  trait AAtSparseSolver extends AAtSolver
  trait AtASolver extends MMUnaryFunc
  trait AtADenseSolver extends AtASolver
  trait AtASparseSolver extends AtASolver

}
