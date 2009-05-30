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
import java.awt.Font;
import java.awt.event.ActionListener;
import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.SwingUtilities;

/**
 * Panel for controlling the execution of the Travelling Salesman applet.
 * Contains controls for starting and stopping the route-finding algorithms
 * and for displaying progress and results. 
 * 
 * <br>
 * The original code is from <b>the Watchmaker project</b>
 * (https://watchmaker.dev.java.net/).
 */
final class ExecutionPanel extends JPanel implements ProgressListener {

    private final JButton startButton;
    private final JTextArea output;
    private final JScrollPane scroller;
    private final JProgressBar progressBar;

    ExecutionPanel()
    {
        super(new BorderLayout());        
        JPanel controlPanel = new JPanel(new BorderLayout());
        startButton = new JButton("Start");
        controlPanel.add(startButton, BorderLayout.WEST);
        progressBar = new JProgressBar(0, 100);
        controlPanel.add(progressBar, BorderLayout.CENTER);
        add(controlPanel, BorderLayout.NORTH);
        output = new JTextArea();
        output.setEditable(false);
        output.setLineWrap(true);
        output.setWrapStyleWord(true);
        output.setFont(new Font("Monospaced", Font.PLAIN, 12));
        scroller = new JScrollPane(output);
        scroller.setBorder(BorderFactory.createTitledBorder("Results"));
        add(scroller, BorderLayout.CENTER);
    }


    /**
     * Adds logic to the start button so that something happens when
     * it is clicked.
     * @param actionListener The action to perform when the button is
     * clicked.
     */
    public void addActionListener(ActionListener actionListener)
    {
        startButton.addActionListener(actionListener);
    }


    /**
     * Updates the position of the progress bar.
     */
    @Override
    public void updateProgress(final double percentComplete)
    {
        SwingUtilities.invokeLater(new Runnable()
        {
            @Override
            public void run()
            {
                progressBar.setValue((int) percentComplete);
            }
        });
    }


    /**
     * Appends the specified text to this panel's text area.
     * @param text The text to append.
     */
    public void appendOutput(String text)
    {
        output.append(text);
    }

    
    @Override
    public void setEnabled(boolean b)
    {
        startButton.setEnabled(b);
        scroller.setEnabled(b);
        super.setEnabled(b);
    }
}
