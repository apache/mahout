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

package org.apache.mahout.cf.taste.hadoop.similarity;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.mahout.common.RandomUtils;

/**
 * modelling a pair of user ratings for an item
 */
public final class CoRating implements Writable {

  private float prefValueX;
  private float prefValueY;

  public CoRating() {
  }

  public CoRating(float prefValueX, float prefValueY) {
    this.prefValueX = prefValueX;
    this.prefValueY = prefValueY;
  }

  public float getPrefValueX() {
    return prefValueX;
  }

  public float getPrefValueY() {
    return prefValueY;
  }

  @Override
  public int hashCode() {
    return RandomUtils.hashFloat(prefValueX) + 31 * RandomUtils.hashFloat(prefValueY);
  }

  @Override
  public boolean equals(Object obj) {
    if (obj instanceof CoRating) {
      CoRating other = (CoRating) obj;
      return prefValueX == other.prefValueX && prefValueY == other.prefValueY;
    }
    return false;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    prefValueX = in.readFloat();
    prefValueY = in.readFloat();
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeFloat(prefValueX);
    out.writeFloat(prefValueY);
  }

}
