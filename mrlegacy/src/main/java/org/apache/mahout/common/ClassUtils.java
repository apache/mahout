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

import java.lang.reflect.InvocationTargetException;

public final class ClassUtils {

  private ClassUtils() {}

  public static <T> T instantiateAs(String classname, Class<T> asSubclassOfClass) {
    try {
      return instantiateAs(Class.forName(classname).asSubclass(asSubclassOfClass), asSubclassOfClass);
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException(e);
    }
  }

  public static <T> T instantiateAs(String classname, Class<T> asSubclassOfClass, Class<?>[] params, Object[] args) {
    try {
      return instantiateAs(Class.forName(classname).asSubclass(asSubclassOfClass), asSubclassOfClass, params, args);
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException(e);
    }
  }

  public static <T> T instantiateAs(Class<? extends T> clazz,
                                    Class<T> asSubclassOfClass,
                                    Class<?>[] params,
                                    Object[] args) {
    try {
      return clazz.asSubclass(asSubclassOfClass).getConstructor(params).newInstance(args);
    } catch (InstantiationException ie) {
      throw new IllegalStateException(ie);
    } catch (IllegalAccessException iae) {
      throw new IllegalStateException(iae);
    } catch (NoSuchMethodException nsme) {
      throw new IllegalStateException(nsme);
    } catch (InvocationTargetException ite) {
      throw new IllegalStateException(ite);
    }
  }


  public static <T> T instantiateAs(Class<? extends T> clazz, Class<T> asSubclassOfClass) {
    try {
      return clazz.asSubclass(asSubclassOfClass).getConstructor().newInstance();
    } catch (InstantiationException ie) {
      throw new IllegalStateException(ie);
    } catch (IllegalAccessException iae) {
      throw new IllegalStateException(iae);
    } catch (NoSuchMethodException nsme) {
      throw new IllegalStateException(nsme);
    } catch (InvocationTargetException ite) {
      throw new IllegalStateException(ite);
    }
  }
}
