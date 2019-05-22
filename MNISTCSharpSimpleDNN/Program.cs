﻿using System;
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
            /*
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
            
            */
            Layer l = new Layer(1, 1);
            l.resetBiases(new FixedValueRV(0));
            l.resetWeights(new FixedValueRV(1));
            l.activationFunction = new NullActivation();

            double[] xs = new double[] { 1, 2, 4, 6, 8 };
            double[] ys = new double[] { 1, 4, 5.2, 6.9, 13.2 };
            double avgx = 0.0;
            double avgy = 0.0;

            for(int i = 0; i < xs.Length; i++)
            {
                avgx += xs[i];
                avgy += ys[i];
            }
            avgx = avgx / xs.Length;
            avgy = avgy / ys.Length;

            double sqsum = 0.0;
            double musum = 0.0;

            for(int i = 0; i < xs.Length; i++)
            {
                double xscomp = xs[i] - avgx;
                musum += xscomp * (ys[i]-avgy);
                sqsum += xscomp * xscomp;
            }
            double beta = musum / sqsum;
            double alpha = avgy - (beta * avgx);
            Console.WriteLine("alpha = " + alpha);
            Console.WriteLine("beta = " + beta);

            for (int k = 0; k < 1000; k++)
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

                if (k % 10 == 0)
                {
                    Console.WriteLine("w = " + l.weights[0, 0] + " b = " + l.biases[0]);
                }
            }

            Console.WriteLine("w = " + l.weights[0, 0]);
            Console.WriteLine("b = " + l.biases[0]);
            Console.WriteLine("done!");
            
        }
    }
}
