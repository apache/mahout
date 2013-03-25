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

package org.apache.mahout.classifier.email;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.util.Locale;
import java.util.regex.Pattern;

/**
 * Convert the labels created by the {@link org.apache.mahout.utils.email.MailProcessor} to one consumable
 * by the classifiers
 */
public class PrepEmailMapper extends Mapper<WritableComparable<?>, VectorWritable, Text, VectorWritable> {

  private static final Pattern DASH_DOT = Pattern.compile("-|\\.");
  private static final Pattern SLASH = Pattern.compile("\\/");

  private boolean useListName = false; //if true, use the project name and the list name in label creation
  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    useListName = Boolean.parseBoolean(context.getConfiguration().get(PrepEmailVectorsDriver.USE_LIST_NAME));
  }

  @Override
  protected void map(WritableComparable<?> key, VectorWritable value, Context context)
    throws IOException, InterruptedException {
    String input = key.toString();
    ///Example: /cocoon.apache.org/dev/200307.gz/001401c3414f$8394e160$1e01a8c0@WRPO
    String[] splits = SLASH.split(input);
    //we need the first two splits;
    if (splits.length >= 3) {
      StringBuilder bldr = new StringBuilder();
      bldr.append(escape(splits[1]));
      if (useListName) {
        bldr.append('_').append(escape(splits[2]));
      }
      context.write(new Text(bldr.toString()), value);
    }

  }
  
  private static String escape(CharSequence value) {
    return DASH_DOT.matcher(value).replaceAll("_").toLowerCase(Locale.ENGLISH);
  }
}
