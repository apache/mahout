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

package org.apache.mahout.common;

public class Pair<Left, Right> implements java.util.Map.Entry<Left, Right> {
  protected Left left;

  protected Right right;

  /**
   * construct a new Pair using left and right value
   * 
   * @param left
   * @param right
   */
  public Pair(Left left, Right right) {
    this.left = left;
    this.right = right;
  }

  /**
   * Get the left value
   * 
   * @return Left
   */
  public Left left() {
    return this.left;
  }

  /**
   * Get the right value
   * 
   * @return Right
   */
  public Right right() {
    return this.right;
  }

  /**
   * returns a swapped Pair with left and right interchanged
   * 
   * @return Pair<Right, Left>
   */
  public Pair<Right, Left> swap() {
    return new Pair<Right, Left>(right, left);
  }

  @Override
  public Left getKey() {
    return this.left;
  }

  @Override
  public Right getValue() {
    return this.right;
  }

  @Override
  public Right setValue(Right value) {
    this.right = value;
    return this.right;
  }

  /**
   * some hashcode for Pair made from left and right hash codes
   */
  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + ((left == null) ? 0 : left.hashCode());
    result = prime * result + ((right == null) ? 0 : right.hashCode());
    return result;
  }

  /**
   * @return boolean true if obj is not null and equals(this)
   */
  @SuppressWarnings("unchecked")
  @Override
  public boolean equals(Object obj) {
    if (this == obj)
      return true;
    if (obj == null)
      return false;
    if (getClass() != obj.getClass())
      return false;
    Pair<Left, Right> other = (Pair<Left, Right>) obj;
    if (left == null) {
      if (other.left != null)
        return false;
    } else if (!left.equals(other.left))
      return false;
    if (right == null) {
      if (other.right != null)
        return false;
    } else if (!right.equals(other.right))
      return false;
    return true;
  }
  
  @Override
  public String toString() {    
    return  left.toString() + "-" + right.toString();
  }

}
