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

package org.apache.mahout.clustering.display;

import java.awt.Graphics;
import java.awt.Graphics2D;
import java.io.File;
import java.io.Writer;

import com.google.common.base.Charsets;
import com.google.common.io.Closeables;
import com.google.common.io.Files;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.clustering.spectral.kmeans.SpectralKMeansDriver;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.ManhattanDistanceMeasure;

public class DisplaySpectralKMeans extends DisplayClustering {

  protected static final String SAMPLES = "samples";
  protected static final String OUTPUT = "output";
  protected static final String TEMP = "tmp";
  protected static final String AFFINITIES = "affinities";

  DisplaySpectralKMeans() {
    initialize();
    setTitle("Spectral k-Means Clusters (>" + (int) (significance * 100) + "% of population)");
  }

  public static void main(String[] args) throws Exception {
    DistanceMeasure measure = new ManhattanDistanceMeasure();
    Path samples = new Path(SAMPLES);
    Path output = new Path(OUTPUT);
    Path tempDir = new Path(TEMP);
    Configuration conf = new Configuration();
    HadoopUtil.delete(conf, samples);
    HadoopUtil.delete(conf, output);

    RandomUtils.useTestSeed();
    DisplayClustering.generateSamples();
    writeSampleData(samples);
    Path affinities = new Path(output, AFFINITIES);
    FileSystem fs = FileSystem.get(output.toUri(), conf);
    if (!fs.exists(output)) {
      fs.mkdirs(output);
    }
    Writer writer = null;
    try {
      writer = Files.newWriter(new File(affinities.toString()), Charsets.UTF_8);
      for (int i = 0; i < SAMPLE_DATA.size(); i++) {
        for (int j = 0; j < SAMPLE_DATA.size(); j++) {
          writer.write(i + "," + j + ',' + measure.distance(SAMPLE_DATA.get(i).get(), SAMPLE_DATA.get(j).get()) + '\n');
        }
      }
    } finally {
      Closeables.close(writer, false);
    }
    int maxIter = 10;
    double convergenceDelta = 0.001;
    SpectralKMeansDriver.run(new Configuration(), affinities, output, SAMPLE_DATA.size(), 3, measure, convergenceDelta,
        maxIter, tempDir, false);
    new DisplaySpectralKMeans();
  }

  @Override
  public void paint(Graphics g) {
    plotClusteredSampleData((Graphics2D) g, new Path(OUTPUT));
  }
}
