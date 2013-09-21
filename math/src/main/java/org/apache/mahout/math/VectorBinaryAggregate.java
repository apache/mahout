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

package org.apache.mahout.math;

import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.set.OpenIntHashSet;

import java.util.Iterator;

/**
 * Abstract class encapsulating different algorithms that perform the Vector operations aggregate().
 * x.aggregte(y, fa, fc), for x and y Vectors and fa, fc DoubleDouble functions:
 * - applies the function fc to every element in x and y, fc(xi, yi)
 * - constructs a result iteratively, r0 = fc(x0, y0), ri = fc(r_{i-1}, fc(xi, yi)).
 * This works essentially like a map/reduce functional combo.
 *
 * The names of variables, methods and classes used here follow the following conventions:
 * The vector being assigned to (the left hand side) is called this or x.
 * The right hand side is called that or y.
 * The aggregating (reducing) function to be applied is called fa.
 * The combining (mapping) function to be applied is called fc.
 *
 * The different algorithms take into account the different characteristics of vector classes:
 * - whether the vectors support sequential iteration (isSequential())
 * - what the lookup cost is (getLookupCost())
 * - what the iterator advancement cost is (getIteratorAdvanceCost())
 *
 * The names of the actual classes (they're nested in VectorBinaryAssign) describe the used for assignment.
 * The most important optimization is iterating just through the nonzeros (only possible if f(0, 0) = 0).
 * There are 4 main possibilities:
 * - iterating through the nonzeros of just one vector and looking up the corresponding elements in the other
 * - iterating through the intersection of nonzeros (those indices where both vectors have nonzero values)
 * - iterating through the union of nonzeros (those indices where at least one of the vectors has a nonzero value)
 * - iterating through all the elements in some way (either through both at the same time, both one after the other,
 *   looking up both, looking up just one).
 *
 * The internal details are not important and a particular algorithm should generally not be called explicitly.
 * The best one will be selected through assignBest(), which is itself called through Vector.assign().
 *
 * See https://docs.google.com/document/d/1g1PjUuvjyh2LBdq2_rKLIcUiDbeOORA1sCJiSsz-JVU/edit# for a more detailed
 * explanation.
 */
public abstract class VectorBinaryAggregate {
  public static final VectorBinaryAggregate[] OPERATIONS = {
    new AggregateNonzerosIterateThisLookupThat(),
    new AggregateNonzerosIterateThatLookupThis(),

    new AggregateIterateIntersection(),

    new AggregateIterateUnionSequential(),
    new AggregateIterateUnionRandom(),

    new AggregateAllIterateSequential(),
    new AggregateAllIterateThisLookupThat(),
    new AggregateAllIterateThatLookupThis(),
    new AggregateAllLoop(),
  };

  /**
   * Returns true iff we can use this algorithm to apply fc to x and y component-wise and aggregate the result using fa.
   */
  public abstract boolean isValid(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc);

  /**
   * Estimates the cost of using this algorithm to compute the aggregation. The algorithm is assumed to be valid.
   */
  public abstract double estimateCost(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc);

  /**
   * Main method that applies fc to x and y component-wise aggregating the results with fa. It returns the result of
   * the aggregation.
   */
  public abstract double aggregate(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc);

  /**
   * The best operation is the least expensive valid one.
   */
  public static VectorBinaryAggregate getBestOperation(Vector x, Vector y, DoubleDoubleFunction fa,
                                                       DoubleDoubleFunction fc) {
    int bestOperationIndex = -1;
    double bestCost = Double.POSITIVE_INFINITY;
    for (int i = 0; i < OPERATIONS.length; ++i) {
      if (OPERATIONS[i].isValid(x, y, fa, fc)) {
        double cost = OPERATIONS[i].estimateCost(x, y, fa, fc);
        if (cost < bestCost) {
          bestCost = cost;
          bestOperationIndex = i;
        }
      }
    }
    return OPERATIONS[bestOperationIndex];
  }

  /**
   * This is the method that should be used when aggregating. It selects the best algorithm and applies it.
   */
  public static double aggregateBest(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
    return getBestOperation(x, y, fa, fc).aggregate(x, y, fa, fc);
  }

  public static class AggregateNonzerosIterateThisLookupThat extends VectorBinaryAggregate {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return fa.isLikeRightPlus() && (fa.isAssociativeAndCommutative() || x.isSequentialAccess())
          && fc.isLikeLeftMult();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return x.getNumNondefaultElements() * x.getIteratorAdvanceCost() * y.getLookupCost();
    }

    @Override
    public double aggregate(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      Iterator<Vector.Element> xi = x.nonZeroes().iterator();
      if (!xi.hasNext()) {
        return 0;
      }
      Vector.Element xe = xi.next();
      double result = fc.apply(xe.get(), y.getQuick(xe.index()));
      while (xi.hasNext()) {
        xe = xi.next();
        result = fa.apply(result, fc.apply(xe.get(), y.getQuick(xe.index())));
      }
      return result;
    }
  }

  public static class AggregateNonzerosIterateThatLookupThis extends VectorBinaryAggregate {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return fa.isLikeRightPlus() && (fa.isAssociativeAndCommutative() || y.isSequentialAccess())
          && fc.isLikeRightMult();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return y.getNumNondefaultElements() * y.getIteratorAdvanceCost() * x.getLookupCost() * x.getLookupCost();
    }

    @Override
    public double aggregate(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      Iterator<Vector.Element> yi = y.nonZeroes().iterator();
      if (!yi.hasNext()) {
        return 0;
      }
      Vector.Element ye = yi.next();
      double result = fc.apply(x.getQuick(ye.index()), ye.get());
      while (yi.hasNext()) {
        ye = yi.next();
        result = fa.apply(result, fc.apply(x.getQuick(ye.index()), ye.get()));
      }
      return result;
    }
  }

  public static class AggregateIterateIntersection extends VectorBinaryAggregate {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return fa.isLikeRightPlus() && fc.isLikeMult() && x.isSequentialAccess() && y.isSequentialAccess();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return Math.min(x.getNumNondefaultElements() * x.getIteratorAdvanceCost(),
          y.getNumNondefaultElements() * y.getIteratorAdvanceCost());
    }

    @Override
    public double aggregate(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      Iterator<Vector.Element> xi = x.nonZeroes().iterator();
      Iterator<Vector.Element> yi = y.nonZeroes().iterator();
      Vector.Element xe = null;
      Vector.Element ye = null;
      boolean advanceThis = true;
      boolean advanceThat = true;
      boolean validResult = false;
      double result = 0;
      while (true) {
        if (advanceThis) {
          if (xi.hasNext()) {
            xe = xi.next();
          } else {
            break;
          }
        }
        if (advanceThat) {
          if (yi.hasNext()) {
            ye = yi.next();
          } else {
            break;
          }
        }
        if (xe.index() == ye.index()) {
          double thisResult = fc.apply(xe.get(), ye.get());
          if (validResult) {
            result = fa.apply(result, thisResult);
          } else {
            result = thisResult;
            validResult = true;
          }
          advanceThis = true;
          advanceThat = true;
        } else {
          if (xe.index() < ye.index()) { // f(x, 0) = 0
            advanceThis = true;
            advanceThat = false;
          } else { // f(0, y) = 0
            advanceThis = false;
            advanceThat = true;
          }
        }
      }
      return result;
    }
  }

  public static class AggregateIterateUnionSequential extends VectorBinaryAggregate {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return fa.isLikeRightPlus() && !fc.isDensifying()
          && x.isSequentialAccess() && y.isSequentialAccess();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return Math.max(x.getNumNondefaultElements() * x.getIteratorAdvanceCost(),
          y.getNumNondefaultElements() * y.getIteratorAdvanceCost());
    }

    @Override
    public double aggregate(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      Iterator<Vector.Element> xi = x.nonZeroes().iterator();
      Iterator<Vector.Element> yi = y.nonZeroes().iterator();
      Vector.Element xe = null;
      Vector.Element ye = null;
      boolean advanceThis = true;
      boolean advanceThat = true;
      boolean validResult = false;
      double result = 0;
      while (true) {
        if (advanceThis) {
          if (xi.hasNext()) {
            xe = xi.next();
          } else {
            xe = null;
          }
        }
        if (advanceThat) {
          if (yi.hasNext()) {
            ye = yi.next();
          } else {
            ye = null;
          }
        }
        double thisResult;
        if (xe != null && ye != null) { // both vectors have nonzero elements
          if (xe.index() == ye.index()) {
            thisResult = fc.apply(xe.get(), ye.get());
            advanceThis = true;
            advanceThat = true;
          } else {
            if (xe.index() < ye.index()) { // f(x, 0)
              thisResult = fc.apply(xe.get(), 0);
              advanceThis = true;
              advanceThat = false;
            } else {
              thisResult = fc.apply(0, ye.get());
              advanceThis = false;
              advanceThat = true;
            }
          }
        } else if (xe != null) { // just the first one still has nonzeros
          thisResult = fc.apply(xe.get(), 0);
          advanceThis = true;
          advanceThat = false;
        } else if (ye != null) { // just the second one has nonzeros
          thisResult = fc.apply(0, ye.get());
          advanceThis = false;
          advanceThat = true;
        } else { // we're done, both are empty
          break;
        }
        if (validResult) {
          result = fa.apply(result, thisResult);
        } else {
          result = thisResult;
          validResult =  true;
        }
      }
      return result;
    }
  }

  public static class AggregateIterateUnionRandom extends VectorBinaryAggregate {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return fa.isLikeRightPlus() && !fc.isDensifying()
          && (fa.isAssociativeAndCommutative() || (x.isSequentialAccess() && y.isSequentialAccess()));
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return Math.max(x.getNumNondefaultElements() * x.getIteratorAdvanceCost() * y.getLookupCost(),
          y.getNumNondefaultElements() * y.getIteratorAdvanceCost() * x.getLookupCost());
    }

    @Override
    public double aggregate(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      OpenIntHashSet visited = new OpenIntHashSet();
      Iterator<Vector.Element> xi = x.nonZeroes().iterator();
      boolean validResult = false;
      double result = 0;
      double thisResult;
      while (xi.hasNext()) {
        Vector.Element xe = xi.next();
        thisResult = fc.apply(xe.get(), y.getQuick(xe.index()));
        if (validResult) {
          result = fa.apply(result, thisResult);
        } else {
          result = thisResult;
          validResult = true;
        }
        visited.add(xe.index());
      }
      Iterator<Vector.Element> yi = y.nonZeroes().iterator();
      while (yi.hasNext()) {
        Vector.Element ye = yi.next();
        if (!visited.contains(ye.index())) {
          thisResult = fc.apply(x.getQuick(ye.index()), ye.get());
          if (validResult) {
            result = fa.apply(result, thisResult);
          } else {
            result = thisResult;
            validResult = true;
          }
        }
      }
      return result;
    }
  }

  public static class AggregateAllIterateSequential extends VectorBinaryAggregate {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return x.isSequentialAccess() && y.isSequentialAccess() && !x.isDense() && !y.isDense();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return Math.max(x.size() * x.getIteratorAdvanceCost(), y.size() * y.getIteratorAdvanceCost());
    }

    @Override
    public double aggregate(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      Iterator<Vector.Element> xi = x.all().iterator();
      Iterator<Vector.Element> yi = y.all().iterator();
      boolean validResult = false;
      double result = 0;
      while (xi.hasNext() && yi.hasNext()) {
        Vector.Element xe = xi.next();
        double thisResult = fc.apply(xe.get(), yi.next().get());
        if (validResult) {
          result = fa.apply(result, thisResult);
        } else {
          result = thisResult;
          validResult = true;
        }
      }
      return result;
    }
  }

  public static class AggregateAllIterateThisLookupThat extends VectorBinaryAggregate {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return (fa.isAssociativeAndCommutative() || x.isSequentialAccess())
          && !x.isDense();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return x.size() * x.getIteratorAdvanceCost() * y.getLookupCost();
    }

    @Override
    public double aggregate(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      Iterator<Vector.Element> xi = x.all().iterator();
      boolean validResult = false;
      double result = 0;
      while (xi.hasNext()) {
        Vector.Element xe = xi.next();
        double thisResult = fc.apply(xe.get(), y.getQuick(xe.index()));
        if (validResult) {
          result = fa.apply(result, thisResult);
        } else {
          result = thisResult;
          validResult = true;
        }
      }
      return result;
    }
  }

  public static class AggregateAllIterateThatLookupThis extends VectorBinaryAggregate {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return (fa.isAssociativeAndCommutative() || y.isSequentialAccess())
          && !y.isDense();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return y.size() * y.getIteratorAdvanceCost() * x.getLookupCost();
    }

    @Override
    public double aggregate(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      Iterator<Vector.Element> yi = y.all().iterator();
      boolean validResult = false;
      double result = 0;
      while (yi.hasNext()) {
        Vector.Element ye = yi.next();
        double thisResult = fc.apply(x.getQuick(ye.index()), ye.get());
        if (validResult) {
          result = fa.apply(result, thisResult);
        } else {
          result = thisResult;
          validResult = true;
        }
      }
      return result;
    }
  }

  public static class AggregateAllLoop extends VectorBinaryAggregate {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return true;
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      return x.size() * x.getLookupCost() * y.getLookupCost();
    }

    @Override
    public double aggregate(Vector x, Vector y, DoubleDoubleFunction fa, DoubleDoubleFunction fc) {
      double result = fc.apply(x.getQuick(0), y.getQuick(0));
      for (int i = 1; i < x.size(); ++i) {
        result = fa.apply(result, fc.apply(x.getQuick(i), y.getQuick(i)));
      }
      return result;
    }
  }
}
