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

package org.apache.mahout.common;

import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;

import com.google.common.base.Charsets;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.util.HelpFormatter;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.util.GenericOptionsParser;

public final class CommandLineUtil {
  
  private CommandLineUtil() { }
  
  public static void printHelp(Group group) {
    HelpFormatter formatter = new HelpFormatter();
    formatter.setGroup(group);
    formatter.print();
  }
 
  /**
   * Print the options supported by {@code GenericOptionsParser}.
   * In addition to the options supported by the job, passed in as the
   * group parameter.
   *
   * @param group job-specific command-line options.
   */
  public static void printHelpWithGenericOptions(Group group) throws IOException {
    new GenericOptionsParser(new Configuration(), new org.apache.commons.cli.Options(), new String[0]);
    PrintWriter pw = new PrintWriter(new OutputStreamWriter(System.out, Charsets.UTF_8), true);
    HelpFormatter formatter = new HelpFormatter();
    formatter.setGroup(group);
    formatter.setPrintWriter(pw);
    formatter.setFooter("Specify HDFS directories while running on hadoop; else specify local file system directories");
    formatter.print();
  }

  public static void printHelpWithGenericOptions(Group group, OptionException oe) throws IOException {
    new GenericOptionsParser(new Configuration(), new org.apache.commons.cli.Options(), new String[0]);
    PrintWriter pw = new PrintWriter(new OutputStreamWriter(System.out, Charsets.UTF_8), true);
    HelpFormatter formatter = new HelpFormatter();
    formatter.setGroup(group);
    formatter.setPrintWriter(pw);
    formatter.setException(oe);
    formatter.print();
  }

}
