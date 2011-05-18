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
import org.apache.mahout.math.Varint;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class FeatureVectorWithRatingWritable implements Writable, Cloneable {

  private int idIndex;
  private Float rating;
  private Vector vector;

  public FeatureVectorWithRatingWritable() {  
  }

  public FeatureVectorWithRatingWritable(int idIndex, Float rating, Vector featureVector) {
    this.idIndex = idIndex;
    this.rating = rating;
    this.vector = featureVector;
  }

  public FeatureVectorWithRatingWritable(int idIndex, Vector featureVector) {
    this.idIndex = idIndex;
    this.vector = featureVector;
  }

  public FeatureVectorWithRatingWritable(int idIndex, float rating) {
    this.idIndex = idIndex;
    this.rating = rating;
  }

  public boolean containsFeatureVector() {
    return vector != null;
  }

  public boolean containsRating() {
    return rating != null;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    Varint.writeUnsignedVarInt(idIndex, out);
    boolean containsRating = containsRating();
    out.writeBoolean(containsRating);
    if (containsRating) {
      out.writeFloat(rating);
    }
    boolean containsFeatureVector = containsFeatureVector();
    out.writeBoolean(containsFeatureVector);
    if (containsFeatureVector) {
      VectorWritable.writeVector(out, vector);      
    }
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    rating = null;
    vector = null;
    idIndex = Varint.readUnsignedVarInt(in);
    boolean containsRating = in.readBoolean();
    if (containsRating) {
      rating = in.readFloat();
    }
    boolean containsFeatureVector = in.readBoolean();
    if (containsFeatureVector) {
      VectorWritable vw = new VectorWritable();
      vw.readFields(in);
      vector = vw.get();
    }
  }

  public int getIDIndex() {
    return idIndex;  
  }

  public Float getRating() {
    return rating;
  }

  public Vector getFeatureVector() {
    return vector;
  }

  @Override
  public boolean equals(Object o) {
    if (!(o instanceof FeatureVectorWithRatingWritable)) {
      return false;
    }
    FeatureVectorWithRatingWritable other = (FeatureVectorWithRatingWritable) o;
    if (idIndex != other.idIndex) {
      return false;
    }
    if (rating != null ? !rating.equals(other.rating) : other.rating != null) {
      return false;
    }
    if (vector != null ? !vector.equals(other.vector) : other.vector != null) {
      return false;
    }
    return true;
  }

  @Override
  public int hashCode() {
    int result = 31 * idIndex + (rating != null ? rating.hashCode() : 0);
    result = 31 * result + (vector != null ? vector.hashCode() : 0);
    return result;
  }

  @Override
  protected FeatureVectorWithRatingWritable clone() {
    return new FeatureVectorWithRatingWritable(idIndex, rating, vector);
  }
}
