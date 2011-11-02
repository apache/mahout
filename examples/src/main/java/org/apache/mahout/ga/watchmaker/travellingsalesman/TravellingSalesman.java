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

package org.apache.mahout.ga.watchmaker.travellingsalesman;

import java.awt.BorderLayout;
import java.awt.Container;
import java.awt.Frame;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.Collection;
import java.util.List;

import javax.swing.JApplet;
import javax.swing.JDialog;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.WindowConstants;

import org.uncommons.swing.SwingBackgroundTask;
import org.uncommons.watchmaker.framework.FitnessEvaluator;

/**
 * Applet for comparing evolutionary and brute force approaches to the Travelling Salesman problem.
 * 
 * The original code is from <b>the Watchmaker project</b> (https://watchmaker.dev.java.net/). <br>
 * This class has been modified to add a main function that runs the JApplet inside a JDialog.
 */
public final class TravellingSalesman extends JApplet {

  private final ItineraryPanel itineraryPanel;
  private final StrategyPanel strategyPanel;
  private final ExecutionPanel executionPanel;
  private final FitnessEvaluator<List<String>> evaluator;
  
  /**
   * Creates the applet and lays out its GUI.
   */
  public TravellingSalesman() {
    DistanceLookup distances = new EuropeanDistanceLookup();
    evaluator = new RouteEvaluator(distances);
    itineraryPanel = new ItineraryPanel(distances.getKnownCities());
    strategyPanel = new StrategyPanel(distances);
    executionPanel = new ExecutionPanel();
    add(itineraryPanel, BorderLayout.WEST);
    Container innerPanel = new JPanel(new BorderLayout());
    innerPanel.add(strategyPanel, BorderLayout.NORTH);
    innerPanel.add(executionPanel, BorderLayout.CENTER);
    add(innerPanel, BorderLayout.CENTER);
    
    executionPanel.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent actionEvent) {
        Collection<String> cities = itineraryPanel.getSelectedCities();
        if (cities.size() < 4) {
          JOptionPane.showMessageDialog(TravellingSalesman.this, "Itinerary must include at least 4 cities.",
            "Error", JOptionPane.ERROR_MESSAGE);
        } else {
          try {
            setEnabled(false);
            TravellingSalesmanStrategy strategy = strategyPanel.getStrategy();
            new TSSwingBackgroundTask(strategy, cities, executionPanel, evaluator).execute();
          } catch (IllegalArgumentException ex) {
            JOptionPane.showMessageDialog(TravellingSalesman.this, ex.getMessage(), "Error",
              JOptionPane.ERROR_MESSAGE);
            setEnabled(true);
          }
        }
      }
    });
    validate();
  }

  /**
   * Toggles whether the controls are enabled for input or not.
   * 
   * @param b
   *          Enables the controls if this flag is true, disables them otherwise.
   */
  @Override
  public void setEnabled(boolean b) {
    itineraryPanel.setEnabled(b);
    strategyPanel.setEnabled(b);
    executionPanel.setEnabled(b);
    super.setEnabled(b);
  }
  
  public static void main(String[] args) {
    JDialog dialog = new JDialog((Frame) null, "Travelling Salesman Frame", true);
    dialog.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
    
    dialog.getContentPane().add(new TravellingSalesman());
    dialog.pack();
    dialog.setLocationRelativeTo(null);
    
    dialog.setVisible(true);
  }

  private final class TSSwingBackgroundTask extends SwingBackgroundTask<List<String>> {

    private long elapsedTime;
    private final TravellingSalesmanStrategy strategy;
    private final Collection<String> cities;
    private final ExecutionPanel executionPanel;
    private final FitnessEvaluator<List<String>> evaluator;

    private TSSwingBackgroundTask(TravellingSalesmanStrategy strategy,
                                  Collection<String> cities,
                                  ExecutionPanel executionPanel,
                                  FitnessEvaluator<List<String>> evaluator) {
      this.strategy = strategy;
      this.cities = cities;
      this.executionPanel = executionPanel;
      this.evaluator = evaluator;
    }

    @Override
    protected List<String> performTask() {
      long startTime = System.currentTimeMillis();
      List<String> result = strategy.calculateShortestRoute(cities, executionPanel);
      elapsedTime = System.currentTimeMillis() - startTime;
      return result;
    }

    @Override
    protected void postProcessing(List<String> result) {
      executionPanel.appendOutput(createResultString(strategy.getDescription(), result,
        evaluator.getFitness(result, null), elapsedTime));
      setEnabled(true);
    }

    /**
     * Helper method for formatting a result as a string for display.
     */
    private String createResultString(String strategyDescription,
                                      List<String> shortestRoute,
                                      double distance,
                                      long elapsedTime) {
      StringBuilder buffer = new StringBuilder(100);
      buffer.append('[');
      buffer.append(strategyDescription);
      buffer.append("]\n");
      buffer.append("ROUTE: ");
      for (String s : shortestRoute) {
        buffer.append(s);
        buffer.append(" -> ");
      }
      buffer.append(shortestRoute.get(0));
      buffer.append('\n');
      buffer.append("TOTAL DISTANCE: ");
      buffer.append(String.valueOf(distance));
      buffer.append("km\n");
      buffer.append("(Search Time: ");
      double seconds = (double) elapsedTime / 1000;
      buffer.append(String.valueOf(seconds));
      buffer.append(" seconds)\n\n");
      return buffer.toString();
    }
  }
}
