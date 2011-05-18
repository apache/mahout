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

package org.apache.mahout.common.iterator;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.Iterator;

import com.google.common.base.Function;
import com.google.common.collect.ForwardingIterator;
import com.google.common.collect.Iterators;

/**
 * An iterator that copies the values in an underlying iterator by finding an appropriate copy constructor.
 */
public final class CopyConstructorIterator<T> extends ForwardingIterator<T> {

  private final Iterator<T> delegate;
  private Constructor<T> constructor;

  public CopyConstructorIterator(Iterator<? extends T> copyFrom) {
    this.delegate = Iterators.transform(
        copyFrom,
        new Function<T,T>() {
          @Override
          public T apply(T from) {
            if (constructor == null) {
              Class<T> elementClass = (Class<T>) from.getClass();
              try {
                constructor = elementClass.getConstructor(elementClass);
              } catch (NoSuchMethodException e) {
                throw new IllegalStateException(e);
              }
            }
            try {
              return constructor.newInstance(from);
            } catch (InstantiationException e) {
              throw new IllegalStateException(e);
            } catch (IllegalAccessException e) {
              throw new IllegalStateException(e);
            } catch (InvocationTargetException e) {
              throw new IllegalStateException(e);
            }
          }
        });
  }

  @Override
  protected Iterator<T> delegate() {
    return delegate;
  }

}
