package org.apache.mahout.cf.taste.example;

import java.io.File;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;

/**
 * This class provides a common implementation for parsing input parameters for
 * all taste examples. Currently they only need the path to the recommendations
 * file as input.
 * 
 * The class is safe to be used in threaded contexts.
 */
public final class TasteOptionParser {

  private TasteOptionParser() {
  }

  /**
   * Parse the given command line arguments.
   * @param args the arguments as given to the application.
   * @return the input file if a file was given on the command line, null otherwise. 
   */
  public static File getRatings(String[] args) throws OptionException {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option inputOpt = obuilder.withLongName("input").withRequired(false)
        .withShortName("i").withArgument(
            abuilder.withName("input").withMinimum(1).withMaximum(1).create())
        .withDescription("The Path for input data directory.").create();

    Option helpOpt = DefaultOptionCreator.helpOption(obuilder);

    Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(
        helpOpt).create();

    Parser parser = new Parser();
    parser.setGroup(group);
    CommandLine cmdLine = parser.parse(args);

    if (cmdLine.hasOption(helpOpt)) {
      CommandLineUtil.printHelp(group);
      return null;
    }

    String prefsFile = cmdLine.getValue(inputOpt).toString();
    return new File(prefsFile);
  }

}
