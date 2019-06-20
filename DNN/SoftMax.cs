using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace DNN
{
    public class SoftMax : ILayer
    {
        public ActivationFunction activationFunction { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
        public Vector<double> ayes { get; set; }
        public Matrix<double> deriv_ayes { get; set; }

        public void activate(Vector<double> prevAyes)
        {
            Vector<double> allthees = Vector<double>.Build.Dense(prevAyes.Count);
            prevAyes.Map(Math.Exp, allthees);

            double sum = allthees.Sum();
            ayes = allthees / sum; //Vector<double>.Build.Dense(prevAyes.Count);
            deriv_ayes = Matrix<double>.Build.Dense(prevAyes.Count, prevAyes.Count);

            for (int i = 0; i < prevAyes.Count; i++)
            {
                for(int j = 0; j < prevAyes.Count; j++)
                {
                    if (i == j)
                    {
                        deriv_ayes[i, j] = (1 - ayes[i]);
                    }
                    else
                    {
                        deriv_ayes[i, j] = -(ayes[i] * ayes[j]);
                    }
                }
            }

        }

        public double L2(Vector<double> expected)
        {
            var l1 = ayes - expected;
            return l1.DotProduct(l1);
        }

        public void resetBiases(IRandomVariable rv)
        {
            throw new NotImplementedException();
        }

        public void resetWeights(IRandomVariable rv)
        {
            throw new NotImplementedException();
        }

        public Vector<double> train(Vector<double> upvals, double training_rate)
        {
            var res = deriv_ayes.Multiply(upvals.ToColumnMatrix()); //upvals.PointwiseMultiply(deriv_ayes);
            return res.Column(0).Multiply(training_rate);
        }
    }
}
