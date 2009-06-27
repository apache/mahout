package org.apache.mahout.utils.clustering;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.commons.cli2.util.HelpFormatter;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.jobcontrol.Job;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.clustering.ClusterBase;

import java.io.IOException;
import java.io.Writer;
import java.io.FileWriter;
import java.io.OutputStreamWriter;


/**
 *
 *
 **/
public class ClusterDumper {
  private transient static Log log = LogFactory.getLog(ClusterDumper.class);
  private static final String LINE_SEP = System.getProperty("line.separator");

  public static void main(String[] args) throws IOException, IllegalAccessException, InstantiationException {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();

    Option seqOpt = obuilder.withLongName("seqFile").withRequired(false).withArgument(
            abuilder.withName("seqFile").withMinimum(1).withMaximum(1).create()).
            withDescription("The Sequence File containing the Clusters").withShortName("s").create();
    Option outputOpt = obuilder.withLongName("output").withRequired(false).withArgument(
            abuilder.withName("output").withMinimum(1).withMaximum(1).create()).
            withDescription("The output file.  If not specified, dumps to the console").withShortName("o").create();
    Option substringOpt = obuilder.withLongName("substring").withRequired(false).withArgument(
            abuilder.withName("substring").withMinimum(1).withMaximum(1).create()).
            withDescription("The number of chars of the asFormatString() to print").withShortName("b").create();
    Option helpOpt = obuilder.withLongName("help").
            withDescription("Print out help").withShortName("h").create();

    Group group = gbuilder.withName("Options").withOption(seqOpt).withOption(outputOpt).withOption(substringOpt).create();

    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);

      if (cmdLine.hasOption(helpOpt)) {

        printHelp(group);
        return;
      }

      if (cmdLine.hasOption(seqOpt)) {
        Path path = new Path(cmdLine.getValue(seqOpt).toString());
        System.out.println("Input Path: " + path);
        JobClient client = new JobClient();
        JobConf conf = new JobConf(Job.class);
        client.setConf(conf);
        FileSystem fs = FileSystem.get(path.toUri(), conf);
        SequenceFile.Reader reader = new SequenceFile.Reader(fs, path, conf);

        Writer writer = null;
        if (cmdLine.hasOption(outputOpt)){
          writer = new FileWriter(cmdLine.getValue(outputOpt).toString());
        } else {
          writer = new OutputStreamWriter(System.out);
        }
        int sub = Integer.MAX_VALUE;
        if (cmdLine.hasOption(substringOpt)) {
          sub = Integer.parseInt(cmdLine.getValue(substringOpt).toString());
        }
        Writable key = (Writable) reader.getKeyClass().newInstance();
        ClusterBase value = (ClusterBase) reader.getValueClass().newInstance();
        while (reader.next(key, value)){
          writer.write(value.getId());
          writer.write(":");
          writer.write("name:" + value.getCenter().getName());
          writer.write(":");
          String fmtStr = value.getCenter().asFormatString();
          writer.write(fmtStr.substring(0, Math.min(sub, fmtStr.length())));
          writer.write(LINE_SEP);
          writer.flush();
        }
        if (cmdLine.hasOption(outputOpt)){
          writer.flush();
          writer.close();
        }
      }

    } catch (OptionException e) {
      log.error("Exception", e);
      printHelp(group);
    }

  }

  private static void printHelp(Group group) {
    HelpFormatter formatter = new HelpFormatter();
    formatter.setGroup(group);
    formatter.print();
  }
}