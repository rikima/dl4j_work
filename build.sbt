name := "dl4j_work"

version := "1.0"

scalaVersion := "2.11.8"

classpathTypes += "maven-plugin"

libraryDependencies ++= Seq()

libraryDependencies += "org.nd4j" % "canova-api" % "0.0.0.15"
libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "0.4-rc3.9"
libraryDependencies += "jfree" % "jfreechart" % "1.0.13"
libraryDependencies += "org.bytedeco" % "javacpp" % "1.2"
libraryDependencies += "org.nd4j" % "nd4j" % "0.4-rc3.9"
libraryDependencies += "org.nd4j" % "nd4j-native" % "0.4-rc3.9" classifier "" classifier "macosx-x86_64"