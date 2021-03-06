﻿/**

The MIT License (MIT)

Copyright (c) 2019 Andrew Douglas.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), 
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

The Software is provided “as is”, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, 
fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other 
liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings 
in the Software.

Line by line description: https://writing.kemitchell.com/2016/09/21/MIT-License-Line-by-Line.html

**/
using System;
using MathNet.Numerics.LinearAlgebra;
using DNN;
using DNN.Activations;
using Utilities.RandomVariables;

namespace MNISTCSharpSimpleDNN
{
    class Program
    {
        static void Main(string[] args)
        {
            /*
            Console.WriteLine("starting simple");
            SimpleData sd = new SimpleData(@"C:\Users\potte\simpledata.txt");
            DNN.DNN dnn = new DNN.DNN();

            var (lable,data) = sd.getTrainingImage();
            while(lable != -1)
            {
                var expect = Vector<double>.Build.Dense(new double[] { 0, 0 });
                expect[lable] = 1;
                var l2 = dnn.train(data, expect, 0.1);

                var lastactivation = dnn.activation;
                string s = "";
                double d = -10;
                for (int ji = 0; ji < lastactivation.Count; ji++)
                {
                    if (lastactivation[ji] > d)
                    {
                        d = lastactivation[ji];
                        s = ji.ToString();
                    }
                }

                Console.WriteLine("given a " + lable + " found: " + s + " L2 = " + l2);
                (lable, data) = sd.getTrainingImage();
            }
            */
            /* */
            Console.WriteLine("Start with MNist.");

            MNISTData mdata = new MNISTData(@"/media/andrewd/New Volume1/Users/potte/Downloads"); //C:\Users\potte\Downloads");
            DNN.DNN dnn = new DNN.DNN(3, new int[] { 28 * 28, 48, 32, 10 });

            int correct = 0;
            int wrong = 0;
            int found = -1;

            for (int i = 0; i < 50000; i++)
            {
                (int label, Vector<double> image) = mdata.getTrainingImage();
                Vector<double> expect = Vector<double>.Build.Dense(new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
                expect[label] = 1.0;
                dnn.train(image, expect, 0.018);

                found = -1;
                double d = -9999;
                var lastactivation = dnn.activation;
                for (int ji = 0; ji < lastactivation.Count; ji++)
                {
                    if (lastactivation[ji] > d)
                    {
                        d = lastactivation[ji];
                        found = ji;
                    }
                }

                if (found == label)
                    correct++;
                else
                    wrong++;

                if (i % 1000 == 0)
                {
                    Console.WriteLine("correct: " + correct + " wrong: " + wrong);
                    correct = 0; wrong = 0;
                }
            }

            correct = 0; wrong = 0;
            MNISTData test = new MNISTData(@"/media/andrewd/New Volume1/Users/potte/Downloads/t10k-images.idx3-ubyte", 
                                           @"/media/andrewd/New Volume1/Users/potte/Downloads/t10k-labels.idx1-ubyte");
            for (int i = 0; i < 10000; i++)
            {
                (int label, Vector<double> image) = mdata.getTrainingImage();
                Vector<double> expect = Vector<double>.Build.Dense(new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
                expect[label] = 1.0;
                dnn.activate(image);
                var lastactivation = dnn.activation;
                found = -1;
                double d = -10;
                for (int ji = 0; ji < lastactivation.Count; ji++)
                {
                    if (lastactivation[ji] > d)
                    {
                        d = lastactivation[ji];
                        found = ji;
                    }
                }
                if (found == label) correct++;
                else wrong++;
            }

            Console.WriteLine("Of 10000 tests, " + correct + " were correct, " + wrong + " were wrong.");

            /*
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
            */
            Console.WriteLine("done!");
            
        }
    }
}
