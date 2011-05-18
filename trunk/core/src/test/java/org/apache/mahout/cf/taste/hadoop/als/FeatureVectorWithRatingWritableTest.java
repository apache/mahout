/**
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

package org.apache.mahout.cf.taste.hadoop.als;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

public class FeatureVectorWithRatingWritableTest extends MahoutTestCase {

  @Test
  public void rating() throws Exception {

    FeatureVectorWithRatingWritable rating = new FeatureVectorWithRatingWritable(1, 3.0f);

    assertTrue(rating.containsRating());
    assertFalse(rating.containsFeatureVector());
    assertEquals(1, rating.getIDIndex());
    assertEquals(3.0f, rating.getRating(), 0.0f);
    assertNull(rating.getFeatureVector());

    FeatureVectorWithRatingWritable clonedRating = recreate(FeatureVectorWithRatingWritable.class, rating);

    assertEquals(rating, clonedRating);
    assertTrue(clonedRating.containsRating());
    assertFalse(clonedRating.containsFeatureVector());
    assertEquals(1, clonedRating.getIDIndex());
    assertEquals(3.0f, clonedRating.getRating(), 0.0f);
    assertNull(clonedRating.getFeatureVector());    
  }

  @Test
  public void featureVector() throws Exception {

    Vector v = new DenseVector(new double[] { 1.5, 2.3, 0.9 });

    FeatureVectorWithRatingWritable featureVector = new FeatureVectorWithRatingWritable(7, v);

    assertFalse(featureVector.containsRating());
    assertTrue(featureVector.containsFeatureVector());
    assertEquals(7, featureVector.getIDIndex());
    assertNull(featureVector.getRating());
    assertEquals(v, featureVector.getFeatureVector());

    FeatureVectorWithRatingWritable clonedFeatureVector =
        recreate(FeatureVectorWithRatingWritable.class, featureVector);

    assertEquals(featureVector, clonedFeatureVector);
    assertFalse(clonedFeatureVector.containsRating());
    assertTrue(clonedFeatureVector.containsFeatureVector());
    assertEquals(7, clonedFeatureVector.getIDIndex());
    assertNull(clonedFeatureVector.getRating());
    assertEquals(v, clonedFeatureVector.getFeatureVector());
  }

  static <T extends Writable> T recreate(Class<T> tClass, T original)
      throws IOException, IllegalAccessException, InstantiationException {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    original.write(new DataOutputStream(out));

    T clone = tClass.newInstance();
    clone.readFields(new DataInputStream(new ByteArrayInputStream(out.toByteArray())));
    return clone;
  }

}
