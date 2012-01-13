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

package org.apache.mahout.clustering.display;

import java.awt.Color;
import java.awt.Font;
import java.awt.Frame;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.geom.AffineTransform;
import java.awt.geom.Line2D;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import javax.swing.Timer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.minhash.HashFactory;
import org.apache.mahout.clustering.minhash.MinHashDriver;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterator;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * <p>
 * This class displays the work of the minimum hash algorithm. There are several
 * parameters which can be used:
 * </p>
 * <p>
 * With the first command line argument you can point the plot type of the
 * algorithm. The possible values are:<br>
 * -p: The algorithm displays the different clusters from the sample data as it
 * highlights the points in the separate clusters in a slide show. It can be
 * paused/resumed with the space key.<br>
 * -l: The algorithm draws the lines between every two points in every cluster.
 * Every cluster is indicated with different, randomly chosen color.<br>
 * -s: The algorithm draws the points from the sample data and maps every
 * cluster to a single character symbol which is displayed near the points which
 * belong to that cluster. However usually one point belongs to more than one
 * cluster and it is not very easy to recognize the clusters with that display.<br>
 * By default the algorithm will plot points highlight slide show.
 * </p>
 * <p>
 * With the second command line argument you can determine the time in seconds
 * in a cluster will be highlighted before the cluster is changed. This option
 * is relevant only when -p is passed as the first command line argument.
 * </p>
 **/
public class DisplayMinHash extends DisplayClustering {

  /**
   * Enumeration of the possible plot types for the {@link DisplayMinHash}
   * program.
   */
  public enum PlotType {
    LINES, POINTS, SYMBOLS
  }

  private static final long serialVersionUID = 1L;
  private transient static Logger log = LoggerFactory
      .getLogger(DisplayMinHash.class);

  private static Map<String, List<Vector>> clusters = new HashMap<String, List<Vector>>();
  private static Iterator<Entry<String, List<Vector>>> currentCluster;
  private static List<Vector> currentClusterPoints;
  private static int updatePeriodTime;
  private static long lastUpdateTime = 0;
  private static boolean isSlideShowOnHold = false;
  private static int symbolsFontSize = 6;

  private PlotType plotType = PlotType.POINTS;

  /**
   * Creates a new instance.
   * 
   * @param type
   *          The chosen {@link PlotType} option.
   */
  public DisplayMinHash(PlotType type) {
    if (type == null) {
      log.error("The PlotType passed should not be null. The program will use the default value - POINTS");
    } else {
      this.plotType = type;
    }
    initialize();
    this.setTitle("Minhash Clusters (>" + (int) (significance * 100)
        + "% of population)");
  }

  /**
   * Draws the clusters in the minimum hash algorithm according to the chosen
   * plot type.
   * 
   * @param g
   *          The {@link Graphics} object used to plot the clusters.
   */
  @Override
  public void paint(Graphics g) {
    plotClusters((Graphics2D) g, plotType);
  }

  private static void plotClusters(Graphics2D g2, PlotType plotType) {
    double sx = (double) res / DS;
    g2.setTransform(AffineTransform.getScaleInstance(sx, sx));
    Font f = new Font("Dialog", Font.PLAIN, symbolsFontSize);
    g2.setFont(f);
    switch (plotType) {
    case LINES:
      plotLines(g2);
      break;
    case SYMBOLS:
      plotSymbols(g2);
      break;
    case POINTS:
      plotPoints(g2);
      break;
    default:
      break;
    }
  }

  private static void plotLines(Graphics2D g2) {
    Random rand = new Random();
    for (Map.Entry<String, List<Vector>> entry : clusters.entrySet()) {
      List<Vector> vecs = entry.getValue();

      g2.setColor(new Color(rand.nextInt()));

      for (int i = 0; i < vecs.size(); i += 2) {
        Vector vec1;
        Vector vec2;
        if (i < vecs.size() - 1) {
          vec1 = vecs.get(i);
          vec2 = vecs.get(i + 1);
        } else {
          // wrap around back to the beginning
          vec1 = vecs.get(i);
          vec2 = vecs.get(0);

        }
        plotLine(g2, vec1, vec2);
      }
    }
  }

  private static void plotSymbols(Graphics2D g2) {
    char symbol = 0;
    Random rand = new Random();
    for (Map.Entry<String, List<Vector>> entry : clusters.entrySet()) {
      List<Vector> vecs = entry.getValue();

      g2.setColor(new Color(rand.nextInt()));
      symbol++;

      for (int i = 0; i < vecs.size(); i++) {
        plotSymbols(g2, vecs.get(i), symbol);
      }
    }
  }

  private static void plotPoints(Graphics2D g2) {
    if (currentCluster == null || !currentCluster.hasNext()) {
      currentCluster = clusters.entrySet().iterator();
    }

    if (System.currentTimeMillis() - lastUpdateTime > updatePeriodTime) {
      plotSampleData((Graphics2D) g2);
      currentClusterPoints = currentCluster.next().getValue();
      lastUpdateTime = System.currentTimeMillis();
    }
    plotSampleData(g2);
    g2.setColor(Color.RED);
    Vector dv = new DenseVector(2).assign(0.03);

    for (int i = 0; i < currentClusterPoints.size(); i++) {
      plotRectangle(g2, currentClusterPoints.get(i), dv);
    }
  }

  private static void plotSymbols(Graphics2D g2, Vector vec, char symbol) {
    double[] flip = { 1, -1 };
    Vector v1 = vec.times(new DenseVector(flip));
    int h = SIZE / 2;
    double x1 = v1.get(0) + h;
    double y1 = v1.get(1) + h;
    g2.drawString(Character.toString(symbol), (int) (x1 * DS), (int) (y1 * DS));
  }

  private static void plotLine(Graphics2D g2, Vector vec1, Vector vec2) {
    double[] flip = { 1, -1 };
    Vector v1 = vec1.times(new DenseVector(flip));
    Vector v2 = vec2.times(new DenseVector(flip));
    int h = SIZE / 2;
    double x1 = v1.get(0) + h;
    double y1 = v1.get(1) + h;
    double x2 = v2.get(0) + h;
    double y2 = v2.get(1) + h;
    g2.draw(new Line2D.Double(x1 * DS, y1 * DS, x2 * DS, y2 * DS));
  }

  /**
   * The entry point to the program.
   * 
   * @param args
   *          The command-line arguments. See {@link DisplayMinHash} for
   *          details.
   * 
   * @throws Exception
   *           Thrown if an error occurs during the execution.
   */
  public static void main(String[] args) throws Exception {
    Path samples = new Path("samples");
    Path output = new Path("output", "minhash");

    PlotType type = determinePlotType(args);
    updatePeriodTime = determineUpdatePeriodTime(args);

    Configuration conf = new Configuration();
    HadoopUtil.delete(conf, samples);
    HadoopUtil.delete(conf, output);
    RandomUtils.useTestSeed();
    generateSamples();
    writeSampleData(samples);
    runMinHash(conf, samples, output);
    loadClusters(output);
    logClusters();
    final Frame f = new DisplayMinHash(type);

    if (type == PlotType.POINTS) {
      Timer timer = new Timer(updatePeriodTime, new ActionListener() {
        public void actionPerformed(ActionEvent e) {
          repaint(f);
        }
      });
      timer.start();
    }

    f.addKeyListener(new KeyListener() {

      @Override
      public void keyTyped(KeyEvent arg0) {
      }

      @Override
      public void keyReleased(KeyEvent arg0) {
      }

      @Override
      public void keyPressed(KeyEvent arg0) {
        if (arg0.getKeyCode() == KeyEvent.VK_SPACE) {
          onSpacePressed();
        }
      }
    });
  }

  private static PlotType determinePlotType(String[] args) {
    PlotType type = PlotType.POINTS;
    if (args.length != 0) {
      if (args[0].equals("-p")) {
        type = PlotType.POINTS;
      } else if (args[0].equals("-l")) {
        type = PlotType.LINES;
      } else if (args[0].equals("-s")) {
        type = PlotType.SYMBOLS;
      } else {
        System.out
            .println("Wrong parameter: -p (plot points); -l (plot lines); -s (plot symbols)");
      }
    }
    return type;
  }

  private static int determineUpdatePeriodTime(String[] args) {
    int updatePeriodTimeInMinutes = 1;
    if (args.length >= 2) {
      try {
        updatePeriodTime = Integer.parseInt(args[1]);
      } catch (Exception e) {
        System.out.println(args[1]
            + " isn't valid integer value. 1 second will be used.");
      }
    }
    return updatePeriodTimeInMinutes * 1000;
  }

  private static void repaint(Frame f) {
    if (!isSlideShowOnHold) {
      f.repaint();
    }

  }

  private static void onSpacePressed() {
    isSlideShowOnHold = !isSlideShowOnHold;
  }

  private static void logClusters() {
    int i = 0;
    for (Map.Entry<String, List<Vector>> entry : clusters.entrySet()) {
      String logStr = "Cluster N:" + ++i + ": ";
      List<Vector> vecs = entry.getValue();
      for (Vector vector : vecs) {
        logStr += vector.get(0);
        logStr += ",";
        logStr += vector.get(1);
        logStr += "; ";
      }
      log.info(logStr);
    }
  }

  protected static void loadClusters(Path output) throws IOException {
    Configuration conf = new Configuration();
    SequenceFileDirIterator<Text, VectorWritable> iterator = new SequenceFileDirIterator<Text, VectorWritable>(
        output, PathType.LIST, PathFilters.partFilter(), null, false, conf);
    while (iterator.hasNext()) {
      Pair<Text, VectorWritable> next = iterator.next();
      String key = next.getFirst().toString();
      List<Vector> list = clusters.get(key);
      if (list == null) {
        list = new ArrayList<Vector>();
        clusters.put(key, list);
      }
      list.add(next.getSecond().get());
    }
    log.info("Loaded: " + clusters.size() + " clusters");
  }

  private static void runMinHash(Configuration conf, Path samples, Path output)
      throws Exception {
    MinHashDriver mhd = new MinHashDriver();

    ToolRunner.run(conf, mhd, new String[] { "--input", samples.toString(),
        "--hashType", HashFactory.HashType.MURMUR3.toString(), "--output",
        output.toString(), "--minVectorSize", "1", "--debugOutput"

    });

  }

}
