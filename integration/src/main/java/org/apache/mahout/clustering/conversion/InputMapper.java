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

package org.apache.mahout.clustering.conversion;

import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.Collection;
import java.util.regex.Pattern;

import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class InputMapper extends Mapper<LongWritable, Text, Text, VectorWritable> {

  private static final Pattern SPACE = Pattern.compile(" ");

  private Constructor<?> constructor;

  @Override
  protected void map(LongWritable key, Text values, Context context) throws IOException, InterruptedException {

    String[] numbers = SPACE.split(values.toString());
    // sometimes there are multiple separator spaces
    Collection<Double> doubles = Lists.newArrayList();
    for (String value : numbers) {
      if (!value.isEmpty()) {
        doubles.add(Double.valueOf(value));
      }
    }
    // ignore empty lines in data file
    if (!doubles.isEmpty()) {
      try {
        Vector result = (Vector) constructor.newInstance(doubles.size());
        int index = 0;
        for (Double d : doubles) {
          result.set(index++, d);
        }
        VectorWritable vectorWritable = new VectorWritable(result);
        context.write(new Text(String.valueOf(index)), vectorWritable);

      } catch (InstantiationException e) {
        throw new IllegalStateException(e);
      } catch (IllegalAccessException e) {
        throw new IllegalStateException(e);
      } catch (InvocationTargetException e) {
        throw new IllegalStateException(e);
      }
    }
  }

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    Configuration conf = context.getConfiguration();
    String vectorImplClassName = conf.get("vector.implementation.class.name");
    try {
      Class<? extends Vector> outputClass = conf.getClassByName(vectorImplClassName).asSubclass(Vector.class);
      constructor = outputClass.getConstructor(int.class);
    } catch (NoSuchMethodException e) {
      throw new IllegalStateException(e);
    } catch (ClassNotFoundException e) {
      throw new IllegalStateException(e);
    }
  }

}
