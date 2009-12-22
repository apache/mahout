package org.apache.mahout.collection_codegen;

/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

import org.apache.maven.plugin.AbstractMojo;
import org.apache.maven.plugin.MojoExecutionException;
import org.apache.maven.project.MavenProject;
import org.apache.maven.shared.model.fileset.FileSet;
import org.apache.maven.shared.model.fileset.util.FileSetManager;
import org.apache.velocity.Template;
import org.apache.velocity.VelocityContext;
import org.apache.velocity.app.VelocityEngine;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;

/**
 * @description Generate java code with Velocity.
 * @goal generate 
 * @phase generate-sources
 * @requiresProject true
 */
public class CodeGenerator extends AbstractMojo {
  
  private static final String[] NO_STRINGS = new String[] { null };
  private static final Charset UTF8 = Charset.forName("utf-8");

  /**
   * @parameter default-value="${basedir}/src/test/java-templates"
   */
  private String testTemplateRoot;
  
  /**
   * Path where the generated sources should be placed
   * 
   * @parameter expression="${cg.outputDirectory}"
   *            default-value="${project.build.directory}/generated-test-sources"
   */
  private File testOutputDirectory;
  
  /**
   * @parameter default-value="${basedir}/src/main/java-templates"
   */
  private String sourceTemplateRoot;
  
  /**
   * Path where the generated sources should be placed
   * 
   * @parameter default-value="${project.build.directory}/generated-sources"
   */
   private File outputDirectory;
  
   /**
    * Comma-separated list of value types.
    * @parameter default-value="byte,char,int,long,float,double"
    * @required
    */
   private String valueTypes;

   /**
    * Comma-separated list of key types.
    * @parameter default-value="byte,char,int,long,float,double"
    * @required
    */
   private String keyTypes;
  
  /**
   * @parameter expression="${project}"
   * @required
   */
  private MavenProject project;
  private VelocityEngine velocityEngine;
  
  public void execute() throws MojoExecutionException {
    File f = outputDirectory;
    
    if (!f.exists()) {
      f.mkdirs();
    }
    
    if (testOutputDirectory != null && !testOutputDirectory.exists()) {
      testOutputDirectory.mkdirs();
    }
    
    velocityEngine = new VelocityEngine();
    // we want to use absolute paths.
    velocityEngine.setProperty("file.resource.loader.path", "/");
    try {
      velocityEngine.init();
    } catch (Exception e) {
      throw new MojoExecutionException("Unable to initialize velocity", e);
    }
    
    if (sourceTemplateRoot != null) {
      runGeneration(sourceTemplateRoot, outputDirectory);
    }
    if (testTemplateRoot != null) {
      runGeneration(testTemplateRoot, testOutputDirectory);
    }
    
    if (project != null && outputDirectory != null && outputDirectory.exists()) {
      project.addCompileSourceRoot(outputDirectory.getAbsolutePath());
  }
  if (project != null && testOutputDirectory != null && testOutputDirectory.exists()) {
      project.addTestCompileSourceRoot(testOutputDirectory.getAbsolutePath());
  }

  }

  private void runGeneration(String thisSourceRoot, File thisOutputDirectory) {
    FileSetManager fileSetManager = new FileSetManager();
    FileSet fileSet = new FileSet();
    fileSet.setDirectory(thisSourceRoot);
    List<String> includes = new ArrayList<String>();
    includes.add("**/*.java.t");
    fileSet.setIncludes(includes);

    String[] includedFiles = fileSetManager.getIncludedFiles(fileSet);
    for (String template : includedFiles) {
      File templateFile = new File(thisSourceRoot, template);
      File thisTemplateOutputDirectory = new File(thisOutputDirectory, templateFile.getParentFile().getPath().substring(fileSet.getDirectory().length()));
      thisTemplateOutputDirectory.mkdirs();
      processOneTemplate(templateFile, thisTemplateOutputDirectory);
    }
  }

  private void processOneTemplate(File templateFile, File thisOutputDirectory) {
    boolean hasKey = templateFile.getName().contains("KeyType");
    boolean hasValue = templateFile.getName().contains("ValueType");
    String[] keys;
    if (hasKey) {
      keys = keyTypes.split(",");
    } else {
      keys = NO_STRINGS;
    }
    String[] values;
    if (hasValue) {
      values = valueTypes.split(",");
    } else {
      values = NO_STRINGS;
    }
    for (String key : keys) {
      for (String value : values) {
        String outputName = templateFile.getName().replaceFirst("\\.java\\.t$", ".java");
        String keyCap = null;
        VelocityContext vc = new VelocityContext();
        if (key != null) {
          keyCap = key.toUpperCase().charAt(0) + key.substring(1);
          outputName = outputName.replaceAll("KeyType", keyCap);
          vc.put("keyType", key);
          vc.put("keyTypeCap", keyCap);
        }
        String valueCap = null;
        if (value != null) {
          valueCap = value.toUpperCase().charAt(0) + value.substring(1);
          outputName = outputName.replaceAll("ValueType", valueCap);
          vc.put("valueType", value);
          vc.put("valueTypeCap", valueCap);
        }
        try {
          Template template = velocityEngine.getTemplate(templateFile.getCanonicalPath());
          File outputFile = new File(thisOutputDirectory, outputName);
          getLog().info("Writing to " + outputFile.getAbsolutePath());
          FileOutputStream fos = new FileOutputStream(outputFile);
          OutputStreamWriter osw = new OutputStreamWriter(fos, UTF8);
          template.merge(vc, osw);
          osw.close();
        } catch (Exception e) {
          getLog().error(e);
        }
      }
    }
  }
}
