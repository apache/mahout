package org.apache.mahout.utils.vectors;
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

import org.apache.mahout.common.FileLineIterator;
import org.apache.mahout.math.Vector;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Iterator;
import java.util.regex.Pattern;

public class VectorHelper {
  private static final Pattern TAB_PATTERN = Pattern.compile("\t");

  private VectorHelper() {
  }


  /**
   * Create a String from a vector that fills in the values with the appropriate value from a dictionary where each the ith entry is the term for the ith vector cell..
   * @param vector
   * @param dictionary The dictionary.  See
   * @return The String
   */
  public static String vectorToString(Vector vector, String [] dictionary){
    StringBuilder bldr = new StringBuilder(2048);
    String name = vector.getName();
    if (name != null && name.length() > 0) {
      bldr.append("Name: ").append(name).append(' ');
    }
    bldr.append("elts: {");
    Iterator<Vector.Element> iter = vector.iterateNonZero();
    boolean first = true;
    while (iter.hasNext()) {
      if (first){
        first = false;
      } else {
        bldr.append(", ");
      }
      Vector.Element elt = (Vector.Element) iter.next();
      bldr.append(elt.index()).append(':').append(dictionary[elt.index()]);

    }
    return bldr.toString();
  }


  /**
   * Read in a dictionary file.  Format is:
   * <pre>term DocFreq Index</pre>
   * @param dictFile
   * @return
   * @throws IOException
   */
  public static String [] loadTermDictionary(File dictFile) throws IOException {
    return loadTermDictionary(new FileInputStream(dictFile));
  }

  /**
   * Read in a dictionary file.  Format is:
   * First line is the number of entries
   * <pre>term DocFreq Index</pre>
   */
  public static String [] loadTermDictionary(InputStream is) throws IOException {
    FileLineIterator it = new FileLineIterator(is);

    int numEntries = Integer.parseInt(it.next());
    //System.out.println(numEntries);
    String [] result = new String[numEntries];

    while (it.hasNext()) {
      String line = it.next();
      if (line.startsWith("#")) {
        continue;
      }
      String[] tokens = TAB_PATTERN.split(line);
      if (tokens.length < 3) {
        continue;
      }
      int index = Integer.parseInt(tokens[2]);//tokens[1] is the doc freq
      result[index] = tokens[0];
    }
    return result;
  }
}
