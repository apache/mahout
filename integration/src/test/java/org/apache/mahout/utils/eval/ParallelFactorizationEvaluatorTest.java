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

package org.apache.mahout.utils.eval;

import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.cf.taste.hadoop.als.eval.ParallelFactorizationEvaluator;
import org.apache.mahout.cf.taste.impl.TasteTestCase;
import org.apache.mahout.math.hadoop.MathHelper;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

public class ParallelFactorizationEvaluatorTest extends TasteTestCase {

  @Test
  public void smallIntegration() throws Exception {

    File pairs = getTestTempFile("pairs.txt");
    File userFeatures = getTestTempFile("userFeatures.seq");
    File itemFeatures = getTestTempFile("itemFeatures.seq");
    File tempDir = getTestTempDir("temp");
    File outputDir = getTestTempDir("out");
    outputDir.delete();

    Configuration conf = new Configuration();
    Path inputPath = new Path(pairs.getAbsolutePath());
    FileSystem fs = FileSystem.get(inputPath.toUri(), conf);

    MathHelper.writeEntries(new double[][] {
        new double[] {  1.5, -2,   0.3 },
        new double[] { -0.7,  2,   0.6 },
        new double[] { -1,    2.5, 3   } }, fs, conf, new Path(userFeatures.getAbsolutePath()));

    MathHelper.writeEntries(new double [][] {
        new double[] {  2.3,  0.5, 0   },
        new double[] {  4.7, -1,   0.2 },
        new double[] {  0.6,  2,   1.3 } }, fs, conf, new Path(itemFeatures.getAbsolutePath()));

    writeLines(pairs, "0,0,3", "2,1,-7", "1,0,-2");

    ParallelFactorizationEvaluator evaluator = new ParallelFactorizationEvaluator();
    evaluator.setConf(conf);
    evaluator.run(new String[] { "--output", outputDir.getAbsolutePath(), "--pairs", pairs.getAbsolutePath(),
        "--userFeatures", userFeatures.getAbsolutePath(), "--itemFeatures", itemFeatures.getAbsolutePath(),
        "--tempDir", tempDir.getAbsolutePath() });

    BufferedReader reader = null;
    try {
      reader = new BufferedReader(new FileReader(new File(outputDir, "rmse.txt")));
      double rmse = Double.parseDouble(reader.readLine());
      assertEquals(0.89342, rmse, EPSILON);
    } finally {
      Closeables.closeQuietly(reader);
    }

  }
}