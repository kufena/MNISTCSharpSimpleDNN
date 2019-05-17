using System;
using MathNet.Numerics.LinearAlgebra;
using DNN;
using DNN.RandomVariables;
using DNN.Activations;

namespace MNISTCSharpSimpleDNN
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Start with MNist.");

            MNISTData mdata = new MNISTData(@"C:\Users\potte\Downloads");
            DNN.DNN dnn = new DNN.DNN(3, new int[] { 28 * 28, 16, 16, 10 });
            
            for(int i = 0; i < 50000; i++)
            {
                (int label, Vector<double> image) = mdata.getTrainingImage();
                Vector<double> expect = Vector<double>.Build.Dense(new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
                expect[label] = 1.0;
                double l2 = dnn.train(image, expect);
                if (i % 1000 == 0)
                    Console.WriteLine("L2 = " + l2);
            }
            
            /*
            Layer l = new Layer(1, 1);
            l.resetBiases(new FixedValueRV(0));
            l.resetWeights(new FixedValueRV(1));
            l.activationFunction = new NullActivation();

            for (int k = 0; k < 10; k++)
            {
                l.activate(Vector<double>.Build.Dense(new double[] { 1 }));
                l.train(Vector<double>.Build.Dense(new double[] { l.ayes[0] - 1 }));

                l.activate(Vector<double>.Build.Dense(new double[] { 2 }));
                l.train(Vector<double>.Build.Dense(new double[] { l.ayes[0] - 4 }));

                l.activate(Vector<double>.Build.Dense(new double[] { 4 }));
                l.train(Vector<double>.Build.Dense(new double[] { l.ayes[0] - 5.2 }));

                l.activate(Vector<double>.Build.Dense(new double[] { 6 }));
                l.train(Vector<double>.Build.Dense(new double[] { l.ayes[0] - 6.9 }));

                l.activate(Vector<double>.Build.Dense(new double[] { 8 }));
                l.train(Vector<double>.Build.Dense(new double[] { l.ayes[0] - 13.2 }));
            }

            Console.WriteLine("w = " + l.weights[0, 0]);
            Console.WriteLine("b = " + l.biases[0]);
            Console.WriteLine("done!");
            */
        }
    }
}
