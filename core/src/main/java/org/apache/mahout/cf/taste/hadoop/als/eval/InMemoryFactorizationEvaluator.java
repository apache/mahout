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

package org.apache.mahout.cf.taste.hadoop.als.eval;

import com.google.common.io.Closeables;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SparseMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.Charset;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

/**
 * <p>Measures the root-mean-squared error of a ratring matrix factorization against a test set.</p>
 *
 * <p>the factorization matrices are read into memory, which makes this job pretty fast, if you get OutOfMemoryErrors,
 * use {@link ParallelFactorizationEvaluator} instead</p>
 *
  * <p>Command line arguments specific to this class are:</p>
 *
 * <ol>
 * <li>--output (path): path where output should go</li>
 * <li>--pairs (path): path containing the test ratings, each line must be userID,itemID,rating</li>
 * <li>--userFeatures (path): path to the user feature matrix</li>
 * <li>--itemFeatures (path): path to the item feature matrix</li>
 * </ol>
 */
public class InMemoryFactorizationEvaluator extends AbstractJob {

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new InMemoryFactorizationEvaluator(), args);
  }

  @Override
  public int run(String[] args) throws Exception {

    addOption("pairs", "p", "path containing the test ratings, each line must be userID,itemID,rating", true);
    addOption("userFeatures", "u", "path to the user feature matrix", true);
    addOption("itemFeatures", "i", "path to the item feature matrix", true);
    addOutputOption();

    Map<String,String> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    Path pairs = new Path(parsedArgs.get("--pairs"));
    Path userFeatures = new Path(parsedArgs.get("--userFeatures"));
    Path itemFeatures = new Path(parsedArgs.get("--itemFeatures"));

    Matrix u = readMatrix(userFeatures);
    Matrix m = readMatrix(itemFeatures);

    FullRunningAverage rmseAvg = new FullRunningAverage();
    FullRunningAverage maeAvg = new FullRunningAverage();
    int pairsUsed = 1;
    Writer writer = new OutputStreamWriter(System.out);
    try {
      for (Preference pref : readProbePreferences(pairs)) {
        int userID = (int) pref.getUserID();
        int itemID = (int) pref.getItemID();

        double rating = pref.getValue();
        double estimate = u.getRow(userID).dot(m.getRow(itemID));
        double err = rating - estimate;
        rmseAvg.addDatum(err * err);
        maeAvg.addDatum(Math.abs(err));
        writer.write("Probe [" + pairsUsed + "], rating of user [" + userID + "] towards item [" + itemID + "], " +
            "[" + rating + "] estimated [" + estimate + "]\n");
        pairsUsed++;
      }
      double rmse = Math.sqrt(rmseAvg.getAverage());
      double mae = maeAvg.getAverage();
      writer.write("RMSE: " + rmse + ", MAE: " + mae + "\n");
    } finally {
      Closeables.closeQuietly(writer);
    }
    return 0;
  }

  private Matrix readMatrix(Path dir) throws IOException {

    Matrix matrix = new SparseMatrix(new int[] { Integer.MAX_VALUE, Integer.MAX_VALUE });

    FileSystem fs = dir.getFileSystem(getConf());
    for (FileStatus seqFile : fs.globStatus(new Path(dir, "part-*"))) {
      Path path = seqFile.getPath();
      SequenceFile.Reader reader = null;
      try {
        reader = new SequenceFile.Reader(fs, path, getConf());
        IntWritable key = new IntWritable();
        VectorWritable value = new VectorWritable();
        while (reader.next(key, value)) {
          int row = key.get();
          Iterator<Vector.Element> elementsIterator = value.get().iterateNonZero();
          while (elementsIterator.hasNext()) {
            Vector.Element element = elementsIterator.next();
            matrix.set(row, element.index(), element.get());
          }
        }
      } finally {
        Closeables.closeQuietly(reader);
      }
    }
    return matrix;
  }

  private List<Preference> readProbePreferences(Path dir) throws IOException {

    List<Preference> preferences = new LinkedList<Preference>();
    FileSystem fs = dir.getFileSystem(getConf());
    for (FileStatus seqFile : fs.globStatus(new Path(dir, "part-*"))) {
      Path path = seqFile.getPath();
      InputStream in = null;
      try  {
        in = fs.open(path);
        BufferedReader reader = new BufferedReader(new InputStreamReader(in, Charset.forName("UTF-8")));
        String line;
        while ((line = reader.readLine()) != null) {
          String[] tokens = TasteHadoopUtils.splitPrefTokens(line);
          long userID = Long.parseLong(tokens[0]);
          long itemID = Long.parseLong(tokens[1]);
          float value = Float.parseFloat(tokens[2]);
          preferences.add(new GenericPreference(userID, itemID, value));
        }
      } finally {
        Closeables.closeQuietly(in);
      }
    }
    return preferences;
  }
}