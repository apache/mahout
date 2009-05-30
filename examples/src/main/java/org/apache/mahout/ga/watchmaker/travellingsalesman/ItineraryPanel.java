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
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JPanel;

/**
 * Component for selecting which cities are to be visited by the
 * travelling salesman.
 * 
 * <br>
 * The original code is from <b>the Watchmaker project</b>
 * (https://watchmaker.dev.java.net/).
 */
final class ItineraryPanel extends JPanel {

    private final Collection<JCheckBox> checkBoxes;
    private final JButton selectAllButton;
    private final JButton clearButton;

    ItineraryPanel(List<String> cities)
    {
        super(new BorderLayout());

        JPanel checkBoxPanel = new JPanel(new GridLayout(0, 1));
        checkBoxes = new ArrayList<JCheckBox>(cities.size());
        for (String city : cities)
        {
            JCheckBox checkBox = new JCheckBox(city, false);
            checkBoxes.add(checkBox);
            checkBoxPanel.add(checkBox);
        }
        add(checkBoxPanel, BorderLayout.CENTER);

        JPanel buttonPanel = new JPanel(new GridLayout(2, 1));
        selectAllButton = new JButton("Select All");
        buttonPanel.add(selectAllButton);
        clearButton = new JButton("Clear Selection");
        buttonPanel.add(clearButton);
        ActionListener buttonListener = new ActionListener()
        {

            @Override
            public void actionPerformed(ActionEvent actionEvent)
            {
                boolean select = actionEvent.getSource() == selectAllButton;
                for (JCheckBox checkBox : checkBoxes)
                {
                    checkBox.setSelected(select);
                }
            }
        };
        selectAllButton.addActionListener(buttonListener);
        clearButton.addActionListener(buttonListener);
        add(buttonPanel, BorderLayout.SOUTH);

        setBorder(BorderFactory.createTitledBorder("Itinerary"));
    }


    /**
     * Returns the cities that have been selected as part of the itinerary.
     */
    public Collection<String> getSelectedCities()
    {
        Set<String> cities = new TreeSet<String>();
        for (JCheckBox checkBox : checkBoxes)
        {
            if (checkBox.isSelected())
            {
                cities.add(checkBox.getText());
            }
        }
        return cities;
    }


    @Override
    public void setEnabled(boolean b)
    {
        for (JCheckBox checkBox : checkBoxes)
        {
            checkBox.setEnabled(b);
        }
        selectAllButton.setEnabled(b);
        clearButton.setEnabled(b);
        super.setEnabled(b);
    }
}
