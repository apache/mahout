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

package org.apache.mahout.utils.vectors;

import com.google.common.base.Function;
import com.google.common.collect.Collections2;
import com.google.common.collect.Lists;
import com.google.common.collect.Ordering;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.lucene.util.PriorityQueue;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.FileLineIterator;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.map.OpenObjectIntHashMap;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.regex.Pattern;

public final class VectorHelper {

  private static final Pattern TAB_PATTERN = Pattern.compile("\t");


  private VectorHelper() {
  }

  public static String vectorToCSVString(Vector vector, boolean namesAsComments) throws IOException {
    Appendable bldr = new StringBuilder(2048);
    vectorToCSVString(vector, namesAsComments, bldr);
    return bldr.toString();
  }

  public static String buildJson(Iterable<Pair<String, Double>> iterable) {
    return buildJson(iterable, new StringBuilder(2048));
  }

  public static String buildJson(Iterable<Pair<String, Double>> iterable, StringBuilder bldr) {
    bldr.append('{');
    for (Pair<String, Double> p : iterable) {
      bldr.append(p.getFirst());
      bldr.append(':');
      bldr.append(p.getSecond());
      bldr.append(',');
    }
    if (bldr.length() > 1) {
      bldr.setCharAt(bldr.length() - 1, '}');
    }
    return bldr.toString();
  }

  public static String vectorToSortedString(Vector vector, String[] dictionary) {
    return vectorToJson(vector, dictionary, Integer.MAX_VALUE, true);
  }

  public static List<Pair<Integer, Double>> topEntries(Vector vector, int maxEntries) {
    PriorityQueue<Pair<Integer, Double>> queue = new TDoublePQ<Integer>(-1, maxEntries);
    Iterator<Vector.Element> it = vector.iterateNonZero();
    while (it.hasNext()) {
      Vector.Element e = it.next();
      queue.insertWithOverflow(Pair.of(e.index(), e.get()));
    }
    List<Pair<Integer, Double>> entries = Lists.newArrayList();
    Pair<Integer, Double> pair;
    while ((pair = queue.pop()) != null) {
      if (pair.getFirst() > -1) {
        entries.add(pair);
      }
    }
    Collections.sort(entries, Ordering.natural().reverse());
    return entries;
  }

  public static List<Pair<Integer, Double>> firstEntries(Vector vector, int maxEntries) {
    List<Pair<Integer, Double>> entries = Lists.newArrayList();
    Iterator<Vector.Element> it = vector.iterateNonZero();
    int i = 0;
    while (it.hasNext() && i++ < maxEntries) {
      Vector.Element e = it.next();
      entries.add(Pair.of(e.index(), e.get()));
    }
    return entries;
  }

  public static List<Pair<String, Double>> toWeightedTerms(Collection<Pair<Integer, Double>> entries,
                                                           final String[] dictionary) {
    if (dictionary != null) {
      return Lists.newArrayList(Collections2.transform(entries,
              new Function<Pair<Integer, Double>, Pair<String, Double>>() {
                @Override
                public Pair<String, Double> apply(Pair<Integer, Double> p) {
                  return Pair.of(dictionary[p.getFirst()], p.getSecond());
                }
              }));
    } else {
      return Lists.newArrayList(Collections2.transform(entries,
              new Function<Pair<Integer, Double>, Pair<String, Double>>() {
                @Override
                public Pair<String, Double> apply(Pair<Integer, Double> p) {
                  return Pair.of(Integer.toString(p.getFirst()), p.getSecond());
                }
              }));
    }
  }

  public static String vectorToJson(Vector vector, String[] dictionary, int maxEntries, boolean sort) {
    return buildJson(toWeightedTerms(sort
            ? topEntries(vector, maxEntries)
            : firstEntries(vector, maxEntries), dictionary));
  }

  public static void vectorToCSVString(Vector vector,
                                       boolean namesAsComments,
                                       Appendable bldr) throws IOException {
    if (namesAsComments && vector instanceof NamedVector) {
      bldr.append('#').append(((NamedVector) vector).getName()).append('\n');
    }
    Iterator<Vector.Element> iter = vector.iterator();
    boolean first = true;
    while (iter.hasNext()) {
      if (first) {
        first = false;
      } else {
        bldr.append(',');
      }
      Vector.Element elt = iter.next();
      bldr.append(String.valueOf(elt.get()));
    }
    bldr.append('\n');
  }

  /**
   * Read in a dictionary file. Format is:
   * <p/>
   * <pre>
   * term DocFreq Index
   * </pre>
   */
  public static String[] loadTermDictionary(File dictFile) throws IOException {
    return loadTermDictionary(new FileInputStream(dictFile));
  }

  /**
   * Read a dictionary in {@link SequenceFile} generated by
   * {@link org.apache.mahout.vectorizer.DictionaryVectorizer}
   *
   * @param filePattern <PATH TO DICTIONARY>/dictionary.file-*
   */
  public static String[] loadTermDictionary(Configuration conf, String filePattern) {
    OpenObjectIntHashMap<String> dict = new OpenObjectIntHashMap<String>();
    for (Pair<Text, IntWritable> record :
            new SequenceFileDirIterable<Text, IntWritable>(new Path(filePattern), PathType.GLOB,
                    null, null, true, conf)) {
      dict.put(record.getFirst().toString(), record.getSecond().get());
    }
    String[] dictionary = new String[dict.size()];
    for (String feature : dict.keys()) {
      dictionary[dict.get(feature)] = feature;
    }
    return dictionary;
  }

  /**
   * Read in a dictionary file. Format is: First line is the number of entries
   * <p/>
   * <pre>
   * term DocFreq Index
   * </pre>
   */
  private static String[] loadTermDictionary(InputStream is) throws IOException {
    FileLineIterator it = new FileLineIterator(is);

    int numEntries = Integer.parseInt(it.next());
    String[] result = new String[numEntries];

    while (it.hasNext()) {
      String line = it.next();
      if (line.startsWith("#")) {
        continue;
      }
      String[] tokens = TAB_PATTERN.split(line);
      if (tokens.length < 3) {
        continue;
      }
      int index = Integer.parseInt(tokens[2]); // tokens[1] is the doc freq
      result[index] = tokens[0];
    }
    return result;
  }

  private static class TDoublePQ<T> extends PriorityQueue<Pair<T, Double>> {
    private final T sentinel;

    private TDoublePQ(T sentinel, int size) {
      super(size);
      this.sentinel = sentinel;
    }

    @Override
    protected boolean lessThan(Pair<T, Double> a,
                               Pair<T, Double> b) {
      return a.getSecond().compareTo(b.getSecond()) < 0;
    }

    @Override
    protected Pair<T, Double> getSentinelObject() {
      return Pair.of(sentinel, Double.NEGATIVE_INFINITY);
    }
  }
}
