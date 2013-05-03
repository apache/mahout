package org.apache.mahout.math;

import java.util.Collection;
import java.util.List;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.function.Functions;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import static org.junit.Assert.assertEquals;

@RunWith(Parameterized.class)
public final class VectorBinaryAggregateTest {
  private static final int CARDINALITY = 10;

  private DoubleDoubleFunction aggregator;
  private DoubleDoubleFunction combiner;
  private VectorBinaryAggregate operation;

  @SuppressWarnings("unchecked")
  @Parameterized.Parameters
  public static Collection<Object[]> generateData() {
    List<Object[]> data = Lists.newArrayList();
    for (List entry : Sets.cartesianProduct(Lists.newArrayList(
        ImmutableSet.of(Functions.PLUS, Functions.PLUS_ABS, Functions.MAX),
        ImmutableSet.of(Functions.PLUS, Functions.PLUS_ABS, Functions.MULT, Functions.MULT_RIGHT_PLUS1,
            Functions.MINUS),
        ImmutableSet.copyOf(VectorBinaryAggregate.operations)))) {
      data.add(entry.toArray());
    }
    return data;
  }

  public VectorBinaryAggregateTest(DoubleDoubleFunction aggregator, DoubleDoubleFunction combiner,
                                   VectorBinaryAggregate operation) {
    this.aggregator = aggregator;
    this.combiner = combiner;
    this.operation = operation;
  }

  @Test
  public void testAll() {
    SequentialAccessSparseVector x = new SequentialAccessSparseVector(CARDINALITY);
    for (int i = 1; i < x.size(); ++i) {
      x.setQuick(i, i);
    }
    SequentialAccessSparseVector y = new SequentialAccessSparseVector(x);

    System.out.printf("aggregator %s; combiner %s; operation %s\n", aggregator, combiner, operation);
    double expectedResult = combiner.apply(0, 0);
    for (int i = 1; i < x.size(); ++i) {
      expectedResult = aggregator.apply(expectedResult, combiner.apply(i, i));
    }

    double result = operation.aggregate(x, y, aggregator, combiner);

    assertEquals(expectedResult, result, 0.0);

  }
}
