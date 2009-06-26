package org.apache.mahout.utils;

import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.util.HelpFormatter;


/**
 *
 *
 **/
public class CommandLineUtil {

  public static void printHelp(Group group) {
    HelpFormatter formatter = new HelpFormatter();
    formatter.setGroup(group);
    formatter.print();
  }

}
