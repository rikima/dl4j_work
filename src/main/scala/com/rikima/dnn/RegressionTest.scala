package com.rikima.dnn

import org.deeplearning4j.datasets.iterator.DataSetIterator
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.examples.feedforward.regression.function._
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener

import org.jfree.chart.ChartFactory
import org.jfree.chart.ChartPanel
import org.jfree.chart.plot.PlotOrientation
import org.jfree.data.xy.XYSeries
import org.jfree.data.xy.XYSeriesCollection

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions

import javax.swing._
import java.util.Collections
import java.util.Random;

/**
  * Created by a14350 on 2016/05/20.
  */
object RegressionTest {
  val seed = 12345;
  //Number of iterations per minibatch
  val iterations = 1
  //Number of epochs (full passes of the data)
  val nEpochs = 2000
  //How frequently should we plot the network output?
  val plotFrequency = 500
  //Number of data pos
  val  nSamples = 1000
  //Batch size: i.e., each epoch has nSamples/batchSize parameter updates
  val batchSize = 100
  //Network learning rate
  val learningRate = 0.01
  val rand = new Random(seed)
  var numInputs = 1
  var numOutputs = 1


  def getDeepDenseLayerNetworkConfiguration(): MultiLayerConfiguration = {
    val numHiddenNodes = 50
    new NeuralNetConfiguration.Builder().seed(seed)
      .iterations(iterations)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(learningRate)
      .weightInit(WeightInit.XAVIER)
      .updater(Updater.NESTEROVS).momentum(0.9)
      .list(3)
      .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
        .activation("tanh")
        .build())
      .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
        .activation("tanh")
        .build())
      .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
        .activation("identity")
        .nIn(numHiddenNodes).nOut(numOutputs).build())
      .pretrain(false).backprop(true).build()
  }


  def getSimpleDenseLayerNetworkConfiguration(): MultiLayerConfiguration = {
    val numHiddenNodes = 20
    new NeuralNetConfiguration.Builder()
      //.seed(seed)
      .iterations(iterations)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(learningRate)
      .weightInit(WeightInit.XAVIER)
      .updater(Updater.NESTEROVS).momentum(0.9)
      .list(2)
      .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
        .activation("tanh")
        .build())
      .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
        .activation("identity")
        .nIn(numHiddenNodes).nOut(numOutputs).build())
      .pretrain(false).backprop(true).build()
  }


  def getTrainingData(x: INDArray,  function: MathFunction, batchSize: Int, rand: Random): DataSetIterator = {
    val y = function.getFunctionValues(x)
    val allData = new DataSet(x,y)

    val list = allData.asList()
    Collections.shuffle(list,rand)
    new ListDataSetIterator(list,batchSize)
  }


  //Plot the data
  def plot(function: MathFunction,  x: INDArray,  y: INDArray,  predicted: Array[INDArray]) {
    val dataSet = new XYSeriesCollection()
    addSeries(dataSet,x,y,"True Function (Labels)")

    for(i <- 0 until predicted.length) {
      addSeries(dataSet,x,predicted(i),String.valueOf(i))
    }

    val chart = ChartFactory.createXYLineChart(
      "Regression Example - " + function.getName(),      // chart title
        "X",                        // x axis label
      function.getName() + "(X)", // y axis label
      dataSet,                    // data
      PlotOrientation.VERTICAL,
      true,                       // include legend
      true,                       // tooltips
      false                       // urls
    )

    val panel = new ChartPanel(chart)

    val f = new JFrame()
    f.add(panel)
    f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
    f.pack()

    f.setVisible(true)
  }


  def addSeries(dataSet:  XYSeriesCollection,  x: INDArray,  y: INDArray, label: String): Unit = {
    val xd = x.data().asDouble()
    val yd = y.data().asDouble()
    val s = new XYSeries(label)
    for (j <- 0 until xd.length) {
      s.add(xd(j),yd(j))
    }
    dataSet.addSeries(s)
  }


  /**
    * main
    */
  def main(args: Array[String]): Unit = {

    //Switch these two options to do different functions with different networks
    val fn = new SinXDivXMathFunction()
    val conf = getDeepDenseLayerNetworkConfiguration()

    //Generate the training data
    val x = Nd4j.linspace(-10,10,nSamples).reshape(nSamples, 1)
    val iterator = getTrainingData(x,fn,batchSize,rand)

    //Create the network
    val net = new MultiLayerNetwork(conf)
    net.init()
    net.setListeners(new ScoreIterationListener(1))


    //Train the network on the full data set, and evaluate in periodically
    val networkPredictions = new Array[INDArray](nEpochs/plotFrequency)
    for (i <- 0 until nEpochs) {
      iterator.reset()
      net.fit(iterator)
      if ((i+1) % plotFrequency == 0) {
        networkPredictions(i/ plotFrequency) = net.output(x, false)
      }
    }

    //Plot the target data and the network predictions
    plot(fn,x,fn.getFunctionValues(x),networkPredictions)
  }
}