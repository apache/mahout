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
import org.apache.mahout.matrix.DenseVector;
import org.apache.mahout.matrix.TimesFunction;
import org.apache.mahout.matrix.Vector;

public class DisplayDirichlet extends Frame {
  private static final long serialVersionUID = 1L;

  protected int res; //screen resolution

  protected static final int ds = 72; //default scale = 72 pixels per inch

  protected static final int size = 8; // screen size in inches

  protected static final List<Vector> sampleData = new ArrayList<Vector>();

  protected static List<Model<Vector>[]> result;

  protected static final double significance = 0.05;

  private static final List<Vector> sampleParams = new ArrayList<Vector>();

  protected static final Color[] colors = { Color.red, Color.orange, Color.yellow, Color.green,
      Color.blue, Color.magenta, Color.lightGray };

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

  public DisplayDirichlet() {
    initialize();
  }

  public void initialize() {
    //Get screen resolution
    res = Toolkit.getDefaultToolkit().getScreenResolution();

    //Set Frame size in inches
    this.setSize(size * res, size * res);
    this.setVisible(true);
    this.setTitle("Dirichlet Process Sample Data");

    //Window listener to terminate program.
    this.addWindowListener(new WindowAdapter() {
      @Override
      public void windowClosing(WindowEvent e) {
        System.exit(0);
      }
    });
  }

  public static void main(String[] args) {
    UncommonDistributions.init("Mahout=Hadoop+ML".getBytes());
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
    for (Vector param : sampleParams) {
      v.set(0, param.get(0));
      v.set(1, param.get(1));
      dv.set(0, param.get(2) * 3);
      dv.set(1, param.get(3) * 3);
      plotEllipse(g2, v, dv);
    }
  }

  public void plotSampleData(Graphics g) {
    Graphics2D g2 = (Graphics2D) g;
    double sx = (double) res / ds;
    g2.setTransform(AffineTransform.getScaleInstance(sx, sx));

    // plot the axes
    g2.setColor(Color.BLACK);
    Vector dv = new DenseVector(2).assign(size / 2);
    plotRectangle(g2, new DenseVector(2).assign(2), dv);
    plotRectangle(g2, new DenseVector(2).assign(-2), dv);

    // plot the sample data
    g2.setColor(Color.DARK_GRAY);
    dv.assign(0.03);
    for (Vector v : sampleData)
      plotRectangle(g2, v, dv);
  }

  /**
   * Plot the points on the graphics context
   * @param g2 a Graphics2D context
   * @param v a Vector of rectangle centers
   * @param dv a Vector of rectangle sizes
   */
  public void plotRectangle(Graphics2D g2, Vector v, Vector dv) {
    int h = size / 2;
    double[] flip = { 1, -1 };
    Vector v2 = v.clone().assign(new DenseVector(flip), new TimesFunction());
    v2 = v2.minus(dv.divide(2));
    double x = v2.get(0) + h;
    double y = v2.get(1) + h;
    g2.draw(new Rectangle2D.Double(x * ds, y * ds, dv.get(0) * ds, dv.get(1)
        * ds));
  }

  /**
   * Plot the points on the graphics context
   * @param g2 a Graphics2D context
   * @param v a Vector of rectangle centers
   * @param dv a Vector of rectangle sizes
   */
  public void plotEllipse(Graphics2D g2, Vector v, Vector dv) {
    int h = size / 2;
    double[] flip = { 1, -1 };
    Vector v2 = v.clone().assign(new DenseVector(flip), new TimesFunction());
    v2 = v2.minus(dv.divide(2));
    double x = v2.get(0) + h;
    double y = v2.get(1) + h;
    g2
        .draw(new Ellipse2D.Double(x * ds, y * ds, dv.get(0) * ds, dv.get(1)
            * ds));
  }

  private static void printModels(List<Model<Vector>[]> results, int significant) {
    int row = 0;
    for (Model<Vector>[] r : results) {
      System.out.print("sample[" + row++ + "]= ");
      for (int k = 0; k < r.length; k++) {
        Model<Vector> model = r[k];
        if (model.count() > significant) {
          System.out.print("m" + k + model.toString() + ", ");
        }
      }
      System.out.println();
    }
    System.out.println();
  }

  public static void generateSamples() {
    generateSamples(400, 1, 1, 3);
    generateSamples(300, 1, 0, 0.5);
    generateSamples(300, 0, 2, 0.1);
  }

  public static void generate2dSamples() {
    generate2dSamples(400, 1, 1, 3, 1);
    generate2dSamples(300, 1, 0, 0.5, 1);
    generate2dSamples(300, 0, 2, 0.1, 0.5);
  }

  /**
   * Generate random samples and add them to the sampleData
   * @param num int number of samples to generate
   * @param mx double x-value of the sample mean
   * @param my double y-value of the sample mean
   * @param sd double standard deviation of the samples
   */
  private static void generateSamples(int num, double mx, double my, double sd) {
    double[] params = { mx, my, sd, sd };
    sampleParams.add(new DenseVector(params));
    System.out.println("Generating " + num + " samples m=[" + mx + ", " + my
        + "] sd=" + sd);
    for (int i = 0; i < num; i++)
      sampleData.add(new DenseVector(new double[] {
          UncommonDistributions.rNorm(mx, sd),
          UncommonDistributions.rNorm(my, sd) }));
  }

  /**
   * Generate random samples and add them to the sampleData
   * @param num int number of samples to generate
   * @param mx double x-value of the sample mean
   * @param my double y-value of the sample mean
   * @param sdx double x-value standard deviation of the samples
   * @param sdy double y-value standard deviation of the samples
   */
  private static void generate2dSamples(int num, double mx, double my,
      double sdx, double sdy) {
    double[] params = { mx, my, sdx, sdy };
    sampleParams.add(new DenseVector(params));
    System.out.println("Generating " + num + " samples m=[" + mx + ", " + my
        + "] sd=[" + sdx + ", " + sdy + ']');
    for (int i = 0; i < num; i++)
      sampleData.add(new DenseVector(new double[] {
          UncommonDistributions.rNorm(mx, sdx),
          UncommonDistributions.rNorm(my, sdy) }));
  }

  public static void generateResults(ModelDistribution<Vector> modelDist) {
    DirichletClusterer<Vector> dc = new DirichletClusterer<Vector>(sampleData,
        modelDist, 1.0, 10, 2, 2);
    result = dc.cluster(20);
    printModels(result, 5);
  }

  public static boolean isSignificant(Model<Vector> model) {
    return (((double) model.count() / sampleData.size()) > significance);
  }

}
