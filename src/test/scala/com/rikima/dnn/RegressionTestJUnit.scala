package com.rikima.dnn

import org.junit.Test
import org.junit.Assert._
import org.nd4j.linalg.factory.Nd4j

/**
 * Created by a14350 on 2016/05/20.
 */
class RegressionTestJUnit {
  @Test
  def testND4J(): Unit = {
    val nd = Nd4j.create(Array(1,2,3,4),Array(2,2,2,3));

    val seed = 1256
    Nd4j.getRandom.setSeed(seed)
  }


  @Test
  def testTest() {
    assertEquals(1, 1)
  }
}