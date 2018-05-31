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

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.Collection;
import java.util.List;
import java.util.Random;

import com.google.common.collect.Lists;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.jet.math.Constants;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import static org.junit.Assert.assertEquals;

@RunWith(Parameterized.class)
public class FunctionTest {
  private static final int NUM_POINTS = 100;

  private final Random random = RandomUtils.getRandom();

  private DoubleDoubleFunction function;
  private String functionName;

  @Parameterized.Parameters
  public static Collection<Object[]> generateData() {
    List<Object[]> data = Lists.newArrayList();
    for (Field field : Functions.class.getDeclaredFields()) {
      if (field.getType().isAssignableFrom(DoubleDoubleFunction.class)
          && Modifier.isStatic(field.getModifiers())
          && !field.getName().equals("SECOND_LEFT_ZERO")) {
        try {
          data.add(new Object[] {field.get(null), field.getName()});
        } catch (IllegalAccessException e) {
          System.out.printf("Couldn't access Functions field %s\n", field.getName());
        }
      }
    }
    return data;
  }

  public FunctionTest(DoubleDoubleFunction function, String functionName) {
    this.function = function;
    this.functionName = functionName;
  }

  @Test
  public void testIsLikeRightPlus() {
    if (!function.isLikeRightPlus()) {
      return;
    }
    for (int i = 0; i < NUM_POINTS; ++i) {
      double x = random.nextDouble();
      assertEquals(functionName, x, function.apply(x, 0), 0);
    }
  }

  @Test
  public void testIsLikeLeftMult() {
    if (!function.isLikeLeftMult()) {
      return;
    }
    for (int i = 0; i < NUM_POINTS; ++i) {
      double y = random.nextDouble();
      assertEquals(functionName, 0, function.apply(0, y), 0);
    }
  }

  @Test
  public void testIsLikeRightMult() {
    if (!function.isLikeRightMult()) {
      return;
    }
    for (int i = 0; i < NUM_POINTS; ++i) {
      double x = random.nextDouble();
      assertEquals(functionName, 0, function.apply(x, 0), 0);
    }
  }

  @Test
  public void testIsCommutative() {
    if (!function.isCommutative()) {
      return;
    }
    for (int i = 0; i < NUM_POINTS; ++i) {
      double x = random.nextDouble();
      double y = random.nextDouble();
      assertEquals(functionName, function.apply(x, y), function.apply(y, x), Constants.EPSILON);
    }
  }

  @Test
  public void testIsAssociative() {
    if (!function.isAssociative()) {
      return;
    }
    for (int i = 0; i < NUM_POINTS; ++i) {
      double x = random.nextDouble();
      double y = random.nextDouble();
      double z = random.nextDouble();
      assertEquals(functionName, function.apply(x, function.apply(y, z)), function.apply(function.apply(x, y), z),
          Constants.EPSILON);
    }
  }

  @Test
  public void testIsDensifying() {
    if (!function.isDensifying()) {
      assertEquals(functionName, 0, function.apply(0, 0), 0);
    }
  }
}
