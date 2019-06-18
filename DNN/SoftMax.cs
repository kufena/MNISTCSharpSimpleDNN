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
        public Vector<double> deriv_ayes { get; set; }

        public void activate(Vector<double> prevAyes)
        {
            double sum = prevAyes.Sum();
            double sumsq = sum * sum;
            ayes = Vector<double>.Build.Dense(prevAyes.Count);
            deriv_ayes = Vector<double>.Build.Dense(prevAyes.Count);

            for (int i = 0; i < prevAyes.Count; i++)
            {
                ayes[i] = prevAyes[i] / sum;
                for(int j = 0; j < prevAyes.Count; j++)
                {
                    if (i == j)
                    {
                        deriv_ayes[i] += (sum - prevAyes[i]) / sumsq; 
                    }
                    else
                    {
                        deriv_ayes[i] += (-prevAyes[i]) / sumsq;
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
            var res = upvals.PointwiseMultiply(deriv_ayes);
            return res;
        }
    }
}
