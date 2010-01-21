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
import org.apache.velocity.context.Context;
import org.codehaus.plexus.util.SelectorUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @description Generate java code with Velocity.
 * @goal generate
 * @phase generate-sources
 * @requiresProject true
 */
public class CodeGenerator extends AbstractMojo {
  
  private static final String[] NO_STRINGS = {null};
  private static final Charset UTF8 = Charset.forName("utf-8");
  private Map<String,String> typeToObjectTypeMap;
  
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
   * Location of manually-mantained files. This plugin won't create a file that
   * would compete with one of these.
   * 
   * @parameter default-value="${basedir}/src/main/java"
   */
  private File doNotReplaceMainDirectory;
  
  /**
   * Location of manually-maintained files. This plugin won't create a file that
   * would compete with one of these.
   * 
   * @parameter default-value="${basedir}/src/test/java"
   */
  private File doNotReplaceTestDirectory;
  
  /**
   * Exclusion patterns -- files NOT to generate.
   * 
   * @parameter
   */
  private String[] mainExcludes;
  
  /**
   * Exclusion patterns -- files NOT to generate.
   * 
   * @parameter
   */
  private String[] testExcludes;
  
  /**
   * Comma-separated list of value types.
   * 
   * @parameter default-value="byte,char,int,short,long,float,double"
   * @required
   */
  private String valueTypes;
  
  /**
   * Comma-separated list of key types.
   * 
   * @parameter default-value="byte,char,int,short,long,float,double"
   * @required
   */
  private String keyTypes;
  
  /**
   * @parameter expression="${project}"
   * @required
   */
  private MavenProject project;
  private VelocityEngine mainVelocityEngine;
  private VelocityEngine testVelocityEngine;
  private FileSetManager fileSetManager;
  
  public CodeGenerator() {
    typeToObjectTypeMap = new HashMap<String,String>();
    typeToObjectTypeMap.put("boolean", "Boolean");
    typeToObjectTypeMap.put("byte", "Byte");
    typeToObjectTypeMap.put("char", "Character");
    typeToObjectTypeMap.put("int", "Integer");
    typeToObjectTypeMap.put("short", "Short");
    typeToObjectTypeMap.put("long", "Long");
    typeToObjectTypeMap.put("float", "Float");
    typeToObjectTypeMap.put("double", "Double");
    fileSetManager = new FileSetManager(getLog());
  }
  
  public void execute() throws MojoExecutionException {
    File f = outputDirectory;
    
    if (!f.exists()) {
      f.mkdirs();
    }
    
    if (testOutputDirectory != null && !testOutputDirectory.exists()) {
      testOutputDirectory.mkdirs();
    }
    
    mainVelocityEngine = new VelocityEngine();
    mainVelocityEngine.setProperty("file.resource.loader.path", sourceTemplateRoot);
    if (testTemplateRoot != null) {
      testVelocityEngine = new VelocityEngine();
      testVelocityEngine.setProperty("file.resource.loader.path", testTemplateRoot);
    }

    
    try {
      mainVelocityEngine.init();
      if (testVelocityEngine != null) {
        testVelocityEngine.init();
      }
    } catch (Exception e) {
      throw new MojoExecutionException("Unable to initialize velocity", e);
    }
    
    if (sourceTemplateRoot != null) {
      runGeneration(
          sourceTemplateRoot, mainVelocityEngine, 
          outputDirectory,
          doNotReplaceMainDirectory, mainExcludes);
    }
    if (testTemplateRoot != null) {
      runGeneration(testTemplateRoot, 
          testVelocityEngine,
          testOutputDirectory,
          doNotReplaceTestDirectory, testExcludes);
    }
    
    if (project != null && outputDirectory != null && outputDirectory.exists()) {
      project.addCompileSourceRoot(outputDirectory.getAbsolutePath());
    }
    if (project != null && testOutputDirectory != null
        && testOutputDirectory.exists()) {
      project.addTestCompileSourceRoot(testOutputDirectory.getAbsolutePath());
    }
    
  }
  
  private void runGeneration(String thisSourceRoot, 
      VelocityEngine engine,
      File thisOutputDirectory,
      File thisDoNotReplaceDirectory, String[] excludes) throws MojoExecutionException {
    FileSet fileSet = new FileSet();
    fileSet.setDirectory(thisSourceRoot);
    List<String> includes = new ArrayList<String>();
    includes.add("**/*.java.t");
    fileSet.setIncludes(includes);
    
    String[] includedFiles = fileSetManager.getIncludedFiles(fileSet);
    for (String template : includedFiles) {
      File templateFile = new File(thisSourceRoot, template);
      String subpath = templateFile.getParentFile().getPath().substring(
          fileSet.getDirectory().length());
      thisOutputDirectory.mkdirs();
      File thisDoNotReplaceFull = new File(thisDoNotReplaceDirectory, subpath);
      processOneTemplate(engine, template, thisOutputDirectory,
          thisDoNotReplaceFull, subpath.substring(1), excludes);
    }
  }
  
  private void processOneTemplate(VelocityEngine engine,
      String template, File thisOutputDirectory,
      File thisDoNotReplaceDirectory, String packageDirectory, String[] excludes) throws MojoExecutionException {
    boolean hasKey = template.contains("KeyType");
    boolean hasValue = template.contains("ValueType");
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
        expandOneTemplate(engine, template, thisOutputDirectory,
            thisDoNotReplaceDirectory, excludes, packageDirectory, key, value);
      }
    }
  }
  
  private void expandOneTemplate(VelocityEngine engine, 
      String templateName, 
      File thisOutputDirectory,
      File thisDoNotReplaceDirectory, String[] excludes,
      String packageDirectory, String key, String value) throws MojoExecutionException {
    String outputName = templateName.replaceFirst("\\.java\\.t$",
        ".java");
    Context vc = new VelocityContext();
    if (key != null) {
      String keyCap = key.toUpperCase().charAt(0) + key.substring(1);
      outputName = outputName.replaceAll("KeyType", keyCap);
      vc.put("keyType", key);
      vc.put("keyTypeCap", keyCap);
      vc.put("keyObjectType", typeToObjectTypeMap.get(key));
      boolean floating = "float".equals(key) || "double".equals(key);
      vc.put("keyTypeFloating", floating ? "true" : "false");
    }
    if (value != null) {
      String valueCap = value.toUpperCase().charAt(0) + value.substring(1);
      outputName = outputName.replaceAll("ValueType", valueCap);
      vc.put("valueType", value);
      vc.put("valueTypeCap", valueCap);
      vc.put("valueObjectType", typeToObjectTypeMap.get(value));
      boolean floating = "float".equals(value) || "double".equals(value);
      vc.put("valueTypeFloating", floating ? "true" : "false");

    }
    File outputFile = new File(thisOutputDirectory, outputName);
    if (thisDoNotReplaceDirectory != null) {
      File dnrf = new File(thisDoNotReplaceDirectory, outputName);
      if (dnrf.exists()) {
        getLog().info("Deferring to " + dnrf.getPath());
        return;
      }
    }
    
    if (excludes != null) {
      for (String exclude : excludes) {
        File excludeFile = new File(packageDirectory, outputName);
        if (SelectorUtils.matchPath(exclude, excludeFile.getPath())) {
          getLog().info("Excluding " + excludeFile.getPath());
          return;
        }
      }
    }
    
    try {
      Template template = engine.getTemplate(templateName);
      getLog().info("Writing to " + outputFile.getAbsolutePath());
      outputFile.getParentFile().mkdirs();
      FileOutputStream fos = new FileOutputStream(outputFile);
      OutputStreamWriter osw = new OutputStreamWriter(fos, UTF8);
      template.merge(vc, osw);
      osw.close();
    } catch (Exception e) {
      getLog().error(e);
      throw new MojoExecutionException("Failed to expand template", e);
    }
  }
}
