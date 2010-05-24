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

package org.apache.mahout.cf.taste.hadoop.item;

import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.GenericWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.VarIntWritable;
import org.apache.mahout.math.VarLongWritable;

public final class TupleWritable extends ArrayWritable {
  
  public static class Field extends GenericWritable {
    
    private static final Class<?>[] CLASSES = {VarIntWritable.class, VarLongWritable.class, DoubleWritable.class,
                                               Text.class};
    
    @Override
    protected Class<? extends Writable>[] getTypes() {
      return (Class<? extends Writable>[]) CLASSES;
    }
    
    public Field() { }
    
    public Field(Writable writable) {
      super.set(writable);
    }
  }
  
  public TupleWritable() {
    super(Field.class);
  }
  
  public TupleWritable(int size) {
    this();
    super.set(new Writable[size]);
  }
  
  public TupleWritable(Writable... writables) {
    this();
    Writable[] fields = new Writable[writables.length];
    int i = 0;
    for (Writable writable : writables) {
      fields[i++] = new Field(writable);
    }
    super.set(fields);
  }
  
  private boolean valid(int idx) {
    Writable[] fields = get();
    int length = fields == null ? 0 : fields.length;
    return (idx >= 0) && (idx < length);
  }
  
  private void allocateCapacity() {
    Writable[] oldFields = get();
    int oldCapacity = oldFields == null ? 0 : oldFields.length;
    int newCapacity = oldCapacity + 1 << 1;
    Writable[] newFields = new Writable[newCapacity];
    if ((oldFields != null) && (oldCapacity > 0)) {
      System.arraycopy(oldFields, 0, newFields, 0, oldFields.length);
    }
    set(newFields);
  }
  
  public void set(int idx, Writable field) {
    if (!valid(idx)) {
      allocateCapacity();
    }
    Writable[] fields = get();
    fields[idx] = new Field(field);
  }
  
  public Field get(int idx) {
    if (!valid(idx)) {
      throw new IllegalArgumentException("Invalid index: " + idx);
    }
    return (Field) get()[idx];
  }
  
  public int getInt(int idx) {
    Field field = get(idx);
    Class<? extends Writable> wrappedClass = field.get().getClass();
    if (wrappedClass.equals(VarIntWritable.class)) {
      return ((VarIntWritable) field.get()).get();
    } else {
      throw new IllegalArgumentException("Not an integer: " + wrappedClass);
    }
  }
  
  public long getLong(int idx) {
    Field field = get(idx);
    Class<? extends Writable> wrappedClass = field.get().getClass();
    if (wrappedClass.equals(VarLongWritable.class)) {
      return ((VarLongWritable) field.get()).get();
    } else {
      throw new IllegalArgumentException("Not a long: " + wrappedClass);
    }
  }
  
  public double getDouble(int idx) {
    Field field = get(idx);
    Class<? extends Writable> wrappedClass = field.get().getClass();
    if (wrappedClass.equals(DoubleWritable.class)) {
      return ((DoubleWritable) field.get()).get();
    } else {
      throw new IllegalArgumentException("Not an double: " + wrappedClass);
    }
  }
  
  public Text getText(int idx) {
    Field field = get(idx);
    Class<? extends Writable> wrappedClass = field.get().getClass();
    if (wrappedClass.equals(Text.class)) {
      return (Text) field.get();
    } else {
      throw new IllegalArgumentException("Not an double: " + wrappedClass);
    }
  }
  
}
