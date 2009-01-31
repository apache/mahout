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

package org.apache.mahout.ga.watchmaker.travellingsalesman;

import java.awt.BorderLayout;
import java.awt.Component;
import java.awt.FlowLayout;
import java.awt.event.ItemEvent;
import java.awt.event.ItemListener;
import java.util.List;

import javax.swing.BorderFactory;
import javax.swing.ButtonGroup;
import javax.swing.DefaultListCellRenderer;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JPanel;
import javax.swing.JRadioButton;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;
import javax.swing.SpringLayout;

import org.uncommons.swing.SpringUtilities;
import org.uncommons.watchmaker.framework.Probability;
import org.uncommons.watchmaker.framework.SelectionStrategy;
import org.uncommons.watchmaker.framework.selection.RankSelection;
import org.uncommons.watchmaker.framework.selection.RouletteWheelSelection;
import org.uncommons.watchmaker.framework.selection.StochasticUniversalSampling;
import org.uncommons.watchmaker.framework.selection.TournamentSelection;
import org.uncommons.watchmaker.framework.selection.TruncationSelection;

/**
 * Panel for configuring a route-finding strategy for the travelling salesman
 * problem.
 * 
 * <br>
 * The original code is from <b>the Watchmaker project</b>
 * (https://watchmaker.dev.java.net/).<br>
 * The <code>EvolutionPanel</code> has been modified to add a "distributed
 * (mahout)" JCheckBox.
 */
final class StrategyPanel extends JPanel {

  private final DistanceLookup distances;

  private final JRadioButton evolutionOption;

  private final JRadioButton bruteForceOption;

  private final EvolutionPanel evolutionPanel;

  /**
   * Creates a panel with components for controlling the route-finding strategy.
   * 
   * @param distances Data used by the strategy in order to calculate shortest
   *        routes.
   */
  StrategyPanel(DistanceLookup distances) {
    super(new BorderLayout());
    this.distances = distances;
    evolutionOption = new JRadioButton("Evolution", true);
    bruteForceOption = new JRadioButton("Brute Force", false);
    evolutionOption.addItemListener(new ItemListener() {
      @Override
      public void itemStateChanged(ItemEvent itemEvent) {
        evolutionPanel.setEnabled(evolutionOption.isSelected());
      }
    });
    ButtonGroup strategyGroup = new ButtonGroup();
    strategyGroup.add(evolutionOption);
    strategyGroup.add(bruteForceOption);
    evolutionPanel = new EvolutionPanel();
    add(evolutionOption, BorderLayout.NORTH);
    add(evolutionPanel, BorderLayout.CENTER);
    add(bruteForceOption, BorderLayout.SOUTH);
    setBorder(BorderFactory.createTitledBorder("Route-Finding Strategy"));
  }

  public TravellingSalesmanStrategy getStrategy() {
    if (bruteForceOption.isSelected()) {
      return new BruteForceTravellingSalesman(distances);
    } else {
      return evolutionPanel.getStrategy();
    }
  }

  @Override
  public void setEnabled(boolean b) {
    evolutionOption.setEnabled(b);
    bruteForceOption.setEnabled(b);
    evolutionPanel.setEnabled(b && evolutionOption.isSelected());
    super.setEnabled(b);
  }

  /**
   * Panel of evolution controls.
   */
  private final class EvolutionPanel extends JPanel {
    private final JLabel populationLabel;

    private final JSpinner populationSpinner;

    private final JLabel elitismLabel;

    private final JSpinner elitismSpinner;

    private final JLabel generationsLabel;

    private final JSpinner generationsSpinner;

    private final JLabel selectionLabel;

    private final JComboBox selectionCombo;

    private final JCheckBox crossoverCheckbox;

    private final JCheckBox mutationCheckbox;

    private final JCheckBox distributedCheckbox;

    EvolutionPanel() {
      super(new FlowLayout(FlowLayout.LEFT, 0, 0));
      JPanel innerPanel = new JPanel(new SpringLayout());

      populationLabel = new JLabel("Population Size: ");
      populationSpinner = new JSpinner(new SpinnerNumberModel(300, 2, 10000, 1));
      populationLabel.setLabelFor(populationSpinner);
      innerPanel.add(populationLabel);
      innerPanel.add(populationSpinner);

      elitismLabel = new JLabel("Elitism: ");
      elitismSpinner = new JSpinner(new SpinnerNumberModel(3, 0, 10000, 1));
      elitismLabel.setLabelFor(elitismSpinner);
      innerPanel.add(elitismLabel);
      innerPanel.add(elitismSpinner);

      generationsLabel = new JLabel("Number of Generations: ");
      generationsSpinner = new JSpinner(
          new SpinnerNumberModel(100, 1, 10000, 1));
      generationsLabel.setLabelFor(generationsSpinner);
      innerPanel.add(generationsLabel);
      innerPanel.add(generationsSpinner);

      selectionLabel = new JLabel("Selection Strategy: ");
      innerPanel.add(selectionLabel);

      SelectionStrategy<?>[] selectionStrategies = {
          new RankSelection(), new RouletteWheelSelection(),
          new StochasticUniversalSampling(), new TournamentSelection(new Probability(0.95d)),
          new TruncationSelection(0.5d)};

      selectionCombo = new JComboBox(selectionStrategies);
      selectionCombo.setRenderer(new DefaultListCellRenderer() {
        @Override
        public Component getListCellRendererComponent(JList list, Object value,
            int index, boolean isSelected, boolean hasFocus) {
          SelectionStrategy<?> strategy = (SelectionStrategy<?>) value;
          String text = strategy.getClass().getSimpleName();
          return super.getListCellRendererComponent(list, text, index,
              isSelected, hasFocus);
        }
      });
      selectionCombo.setSelectedIndex(selectionCombo.getItemCount() - 1);
      innerPanel.add(selectionCombo);

      crossoverCheckbox = new JCheckBox("Cross-over", true);
      mutationCheckbox = new JCheckBox("Mutation", true);
      distributedCheckbox = new JCheckBox("Distributed (Mahout)", false);

      innerPanel.add(crossoverCheckbox);
      innerPanel.add(mutationCheckbox);
      innerPanel.add(distributedCheckbox);
      innerPanel.add(new JLabel()); // pour avoir un nombre paire de components

      SpringUtilities.makeCompactGrid(innerPanel, 6, 2, 30, 6, 6, 6);
      add(innerPanel);
    }

    @Override
    public void setEnabled(boolean b) {
      populationLabel.setEnabled(b);
      populationSpinner.setEnabled(b);
      elitismLabel.setEnabled(b);
      elitismSpinner.setEnabled(b);
      generationsLabel.setEnabled(b);
      generationsSpinner.setEnabled(b);
      selectionLabel.setEnabled(b);
      selectionCombo.setEnabled(b);
      crossoverCheckbox.setEnabled(b);
      mutationCheckbox.setEnabled(b);
      distributedCheckbox.setEnabled(b);
      super.setEnabled(b);
    }

    @SuppressWarnings("unchecked")
    public TravellingSalesmanStrategy getStrategy() {
      return new EvolutionaryTravellingSalesman(distances,
          (SelectionStrategy<? super List<String>>) selectionCombo
              .getSelectedItem(), (Integer) populationSpinner.getValue(),
          (Integer) elitismSpinner.getValue(), (Integer) generationsSpinner
              .getValue(), crossoverCheckbox.isSelected(), mutationCheckbox
              .isSelected(), distributedCheckbox.isSelected());
    }
  }
}
