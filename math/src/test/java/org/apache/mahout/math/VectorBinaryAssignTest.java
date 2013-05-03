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
public final class VectorBinaryAssignTest {
  private static final int CARDINALITY = 10;

  private DoubleDoubleFunction function;
  private VectorBinaryAssign operation;

  @SuppressWarnings("unchecked")
  @Parameterized.Parameters
  public static Collection<Object[]> generateData() {
    List<Object[]> data = Lists.newArrayList();
    for (List entry : Sets.cartesianProduct(Lists.newArrayList(
        ImmutableSet.of(Functions.PLUS, Functions.PLUS_ABS, Functions.MULT, Functions.MULT_RIGHT_PLUS1,
            Functions.MINUS),
        ImmutableSet.copyOf(VectorBinaryAssign.operations)))) {
      data.add(entry.toArray());
    }
    return data;
  }

  public VectorBinaryAssignTest(DoubleDoubleFunction function, VectorBinaryAssign operation) {
    this.function = function;
    this.operation = operation;
  }

  @Test
  public void testAll() {
    SequentialAccessSparseVector x = new SequentialAccessSparseVector(CARDINALITY);
    for (int i = 0; i < x.size(); ++i) {
      x.setQuick(i, i);
    }
    SequentialAccessSparseVector y = new SequentialAccessSparseVector(x);

    System.out.printf("function %s; operation %s\n", function, operation);

    operation.assign(x, y, function);

    for (int i = 0; i < x.size(); ++i) {
      assertEquals(x.getQuick(i), function.apply(i, i), 0.0);
    }
  }
}
