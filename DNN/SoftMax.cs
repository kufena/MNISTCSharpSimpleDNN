/**

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
            //var res = deriv_ayes.Multiply(upvals.ToColumnMatrix()); //upvals.PointwiseMultiply(deriv_ayes);
            //return res.Column(0).Multiply(training_rate);

            return upvals;
        }
    }
}
