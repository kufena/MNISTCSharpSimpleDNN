using NUnit.Framework;
using DNN;
using DNN.Activations;
using MathNet.Numerics.LinearAlgebra;
using Utilities.RandomVariables;

namespace Tests
{
    public class Tests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void Test1()
        {
            FixedValueRV rf = new FixedValueRV(0.5d);
            NullActivation na = new DNN.Activations.NullActivation();
            Layer layer = new DNN.Layer(3, 3);
            layer.activationFunction = na;
            layer.resetBiases(rf);
            layer.resetWeights(rf);
            Vector<double> v = Vector<double>.Build.Dense(new double[] { 1.0, 2.0, 3.0 });
            Vector<double> expected = Vector<double>.Build.Dense(new double[] { 3.5, 3.5, 3.5 });
            layer.activate(v);
            double l2 = layer.L2(expected);
            Assert.AreEqual(0.0, l2, 0.00000001d);
        }

        [Test]
        public void Test2()
        {
            FixedValueRV rf = new FixedValueRV(0.5d);
            NullActivation na = new DNN.Activations.NullActivation();
            Layer layer = new DNN.Layer(3, 3);
            layer.activationFunction = na;
            layer.resetBiases(rf);
            layer.resetWeights(rf);
            Vector<double> v = Vector<double>.Build.Dense(new double[] { 1.0, 2.0, 3.0 });
            Vector<double> expected = Vector<double>.Build.Dense(new double[] { 0,1,0 });
            layer.activate(v);
            double l2 = layer.L2(expected);
            Assert.AreEqual(0.0, l2, 0.00000001d);
            
            var errs = layer.ayes.Subtract(expected);
            layer.train(errs,0.01);
        }
    }
}