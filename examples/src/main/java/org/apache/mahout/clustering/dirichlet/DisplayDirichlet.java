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

package org.apache.mahout.clustering.dirichlet;

import java.awt.Color;
import java.awt.Frame;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Toolkit;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.geom.AffineTransform;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Rectangle2D;
import java.util.ArrayList;
import java.util.List;

import org.apache.mahout.clustering.dirichlet.models.Model;
import org.apache.mahout.clustering.dirichlet.models.ModelDistribution;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DisplayDirichlet extends Frame {
  
  private static final Logger log = LoggerFactory.getLogger(DisplayDirichlet.class);
  
  private static final List<Vector> SAMPLE_PARAMS = new ArrayList<Vector>();
  
  protected static final int DS = 72; // default scale = 72 pixels per inch
  
  protected static final int SIZE = 8; // screen size in inches
  
  protected static final List<VectorWritable> SAMPLE_DATA = new ArrayList<VectorWritable>();
  
  protected static final double SIGNIFICANCE = 0.05;
  
  protected static final Color[] COLORS = {Color.red, Color.orange, Color.yellow, Color.green, Color.blue,
                                           Color.magenta, Color.lightGray};
  
  protected static List<Model<VectorWritable>[]> result;
  
  protected int res; // screen resolution
  
  protected static int k = 12;
  
  public DisplayDirichlet() {
    initialize();
  }
  
  public void initialize() {
    // Get screen resolution
    res = Toolkit.getDefaultToolkit().getScreenResolution();
    
    // Set Frame size in inches
    this.setSize(SIZE * res, SIZE * res);
    this.setVisible(true);
    this.setTitle("Dirichlet Process Sample Data");
    
    // Window listener to terminate program.
    this.addWindowListener(new WindowAdapter() {
      @Override
      public void windowClosing(WindowEvent e) {
        System.exit(0);
      }
    });
  }
  
  public static void main(String[] args) throws Exception {
    RandomUtils.useTestSeed();
    generateSamples();
    new DisplayDirichlet();
  }
  
  // Override the paint() method
  @Override
  public void paint(Graphics g) {
    Graphics2D g2 = (Graphics2D) g;
    plotSampleData(g);
    Vector v = new DenseVector(2);
    Vector dv = new DenseVector(2);
    g2.setColor(Color.RED);
    int i = 0;
    for (Vector param : SAMPLE_PARAMS) {
      i++;
      v.set(0, param.get(0));
      v.set(1, param.get(1));
      dv.set(0, param.get(2) * 3);
      dv.set(1, param.get(3) * 3);
      plotEllipse(g2, v, dv);
    }
  }
  
  public void plotSampleData(Graphics g) {
    Graphics2D g2 = (Graphics2D) g;
    double sx = (double) res / DS;
    g2.setTransform(AffineTransform.getScaleInstance(sx, sx));
    
    // plot the axes
    g2.setColor(Color.BLACK);
    Vector dv = new DenseVector(2).assign(SIZE / 2.0);
    plotRectangle(g2, new DenseVector(2).assign(2), dv);
    plotRectangle(g2, new DenseVector(2).assign(-2), dv);
    
    // plot the sample data
    g2.setColor(Color.DARK_GRAY);
    dv.assign(0.03);
    for (VectorWritable v : SAMPLE_DATA) {
      plotRectangle(g2, v.get(), dv);
    }
  }
  
  /**
   * Draw a rectangle on the graphics context
   * 
   * @param g2
   *          a Graphics2D context
   * @param v
   *          a Vector of rectangle center
   * @param dv
   *          a Vector of rectangle dimensions
   */
  public static void plotRectangle(Graphics2D g2, Vector v, Vector dv) {
    double[] flip = {1, -1};
    Vector v2 = v.times(new DenseVector(flip));
    v2 = v2.minus(dv.divide(2));
    int h = SIZE / 2;
    double x = v2.get(0) + h;
    double y = v2.get(1) + h;
    g2.draw(new Rectangle2D.Double(x * DS, y * DS, dv.get(0) * DS, dv.get(1) * DS));
  }
  
  /**
   * Draw an ellipse on the graphics context
   * 
   * @param g2
   *          a Graphics2D context
   * @param v
   *          a Vector of ellipse center
   * @param dv
   *          a Vector of ellipse dimensions
   */
  public static void plotEllipse(Graphics2D g2, Vector v, Vector dv) {
    double[] flip = {1, -1};
    Vector v2 = v.times(new DenseVector(flip));
    v2 = v2.minus(dv.divide(2));
    int h = SIZE / 2;
    double x = v2.get(0) + h;
    double y = v2.get(1) + h;
    g2.draw(new Ellipse2D.Double(x * DS, y * DS, dv.get(0) * DS, dv.get(1) * DS));
  }
  
  private static void printModels(List<Model<VectorWritable>[]> results, int significant) {
    int row = 0;
    StringBuilder models = new StringBuilder();
    for (Model<VectorWritable>[] r : results) {
      models.append("sample[").append(row++).append("]= ");
      for (int k = 0; k < r.length; k++) {
        Model<VectorWritable> model = r[k];
        if (model.count() > significant) {
          models.append('m').append(k).append(model).append(", ");
        }
      }
      models.append('\n');
    }
    models.append('\n');
    log.info(models.toString());
  }
  
  public static void generateSamples() {
    generateSamples(500, 1, 1, 3);
    generateSamples(300, 1, 0, 0.5);
    generateSamples(300, 0, 2, 0.1);
  }
  
  public static void generate2dSamples() {
    generate2dSamples(500, 1, 1, 3, 1);
    generate2dSamples(300, 1, 0, 0.5, 1);
    generate2dSamples(300, 0, 2, 0.1, 0.5);
  }
  
  /**
   * Generate random samples and add them to the sampleData
   * 
   * @param num
   *          int number of samples to generate
   * @param mx
   *          double x-value of the sample mean
   * @param my
   *          double y-value of the sample mean
   * @param sd
   *          double standard deviation of the samples
   */
  private static void generateSamples(int num, double mx, double my, double sd) {
    double[] params = {mx, my, sd, sd};
    SAMPLE_PARAMS.add(new DenseVector(params));
    log.info("Generating {} samples m=[{}, {}] sd={}", new Object[] {num, mx, my, sd});
    for (int i = 0; i < num; i++) {
      SAMPLE_DATA.add(new VectorWritable(new DenseVector(new double[] {UncommonDistributions.rNorm(mx, sd),
          UncommonDistributions.rNorm(my, sd)})));
    }
  }
  
  /**
   * Generate random samples and add them to the sampleData
   * 
   * @param num
   *          int number of samples to generate
   * @param mx
   *          double x-value of the sample mean
   * @param my
   *          double y-value of the sample mean
   * @param sdx
   *          double x-value standard deviation of the samples
   * @param sdy
   *          double y-value standard deviation of the samples
   */
  private static void generate2dSamples(int num, double mx, double my, double sdx, double sdy) {
    double[] params = {mx, my, sdx, sdy};
    SAMPLE_PARAMS.add(new DenseVector(params));
    log.info("Generating {} samples m=[{}, {}] sd=[{}, {}]", new Object[] {num, mx, my, sdx, sdy});
    for (int i = 0; i < num; i++) {
      SAMPLE_DATA
          .add(new VectorWritable(new DenseVector(new double[] {UncommonDistributions.rNorm(mx, sdx),
                                                                UncommonDistributions.rNorm(my, sdy)})));
    }
  }
  
  public static void generateResults(ModelDistribution<VectorWritable> modelDist) {
    DirichletClusterer<VectorWritable> dc = new DirichletClusterer<VectorWritable>(SAMPLE_DATA, modelDist,
        1.0, k, 2, 2);
    result = dc.cluster(20);
    printModels(result, 5);
  }
  
  public static boolean isSignificant(Model<VectorWritable> model) {
    return (double) model.count() / SAMPLE_DATA.size() > SIGNIFICANCE;
  }
  
}
