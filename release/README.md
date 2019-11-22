This directory id for release related scrips and resources
=========
`Settings.xml`:
A setting file needed in the `/.m2/` directory In order to sign artifacts
All lines which need to be replace are marked with an  \<!-- EDIT THIS \--> comment.
-----------------
```XML
<settings xmlns="http://maven.apache.org/SETTINGS/1.1.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://maven.apache.org/SETTINGS/1.1.0 http://maven.apache.org/xsd/settings-1.1.0.xsd">
<!-- ASF 2.0 LICENSE -->

 <localRepository>${user.home}/.m2/repository</localRepository>
 <interactiveMode>true</interactiveMode>
 <offline>false</offline>

  <!-- To publish a release for staging to NEXUS via Maven -->
  <profiles>
     <profile>
  <id>apache-release</id>
  <properties>
    <gpg.keyname>THE_EIGHT_DIGIT_GPG_SIGNING_KEY_NAME</gpg.keyname>
    <gpg.passphrase>PASSPHRASE_TO_YOUR_SIGNING_KEY</gpg.passphrase>
    <deploy.altRepository>mahout.releases::default::https://repository.apache.org/service/local/staging/deploy/maven2/</deploy.altRepository>
         <url>https://repository.apache.org/service/local/staging/deploy/maven2/</url>
       </properties>
      </profile>

      <profile>
       <activation>
         <activeByDefault>true</activeByDefault>
       </activation>
      <properties>
          <gpg.keyname>THE_EIGHT_DIGIT_GPG_SIGNING_KEY_NAME</gpg.keyname>
      </properties>
     </profile>
   </profiles>
   <servers>
     <!-- To publish a snapshot to Maven -->
     <server>
       <id>apache.snapshots.https</id>
       <id>website</id>
       <username>ASF_LDAP_USERNAME</username>                               <!-- EDIT THIS -->
       <password>ASF_LDAP_PASSWORD</password>                               <!-- EDIT THIS -->
    </server>
  
     <!-- To publish a website of some part of Maven -->
     <server>
       <id>apache.website</id>
       <username>ASF_LDAP_USRNAME</username>                                 <!-- EDIT THIS -->
       <password>ASF_LDAP_PASSWORD</password>                                <!-- EDIT THIS -->
       <filePermissions>664</filePermissions>
       <directoryPermissions>775</directoryPermissions>
     </server>

     <!-- To stage a release via Maven -->
     <server>
       <id>apache.releases.https</id>
       <username>ASF_LDAP_USERNAME</username>
       <password>ASF_LDAP_PASSWORD</password>                                <!-- EDIT THIS -->
       <privateKey>${user.home}/.ssh/id_rsa</privateKey>
       <passphrase>PASSWORD_TO_YOUR_PRIVATE_SSH_KEY</passphrase>             <!-- EDIT THIS -->
       <filePermissions>664</filePermissions>
       <directoryPermissions>775</directoryPermissions>
       <configuration></configuration>
     </server>
     
     <server>
      <id>THE_EIGHT_DIGIT_GPG_SIGNING_KEY_NAME</id>
      <passphrase>PASSPHRASE_TO_YOUR_SIGNING_KEY</passphrase>            <!-- EDIT THIS -->
     </server>	 

     <!-- To stage a website of some part of Maven -->
     <server>
       <id>apache.website</id> <!-- must match  repository identifier in site:stage-deploy -->
       <username>ASF_LDAP_USERNAME</username>                               <!-- EDIT THIS -->
       <filePermissions>664</filePermissions>
       <directoryPermissions>775</directoryPermissions>
     </server>
   </servers>
</settings> 
```

Required settings for a release via `maven 3.3.9` are:

```md
ASF_LDAP_USERNAME
ASF_LDAP_PASSWORD

```
in order to create an scp ssession with the repository to upload artifact in the `deploy` phase.

```md
THE_EIGHT_DIGIT_GPG_SIGNING_KEY_NAME
PASSPHRASE_TO_YOUR_SIGNING_KEY
```

Are used to sign and verify commits.  Theese are required variables which are used when creating a mahout release (and only if you inted to release.).
The keyname: `THE_EIGHT_DIGIT_GPG_SIGNING_KEY_NAME` for lack of any imagination listed here must be recognized by ASF, and listed in the mahout `/Distribution/KEYS` file.

Edit these variables to deploy to an other URL: 
```xml
         <deploy.altRepository>mahout.releases::default::https://repository.apache.org/service/local/staging/deploy/maven2/</deploy.altRepository>
         <url>https://repository.apache.org/service/local/staging/deploy/maven2/</url>
```
