using System;
using MathNet.Numerics.LinearAlgebra;
using DNN;

namespace MNISTCSharpSimpleDNN
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Start with MNist.");

            MNISTData mdata = new MNISTData(@"C:\Users\potte\Downloads");
            DNN.DNN dnn = new DNN.DNN(3, new int[] { 28 * 28, 200, 100, 10 });
            
            for(int i = 0; i < 1000; i++)
            {
                (int label, Vector<double> image) = mdata.getTrainingImage();
                Vector<double> expect = Vector<double>.Build.Dense(new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
                expect[label] = 1.0;
                double l2 = dnn.train(image, expect);
                if (i % 100 == 0)
                    Console.WriteLine("L2 = " + l2);
            }
        }
    }
}
