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
using Utilities.RandomVariables;

namespace DNN
{
    public class Layer : ILayer
    {

        int inputs { get; set; }
        int outputs { get; set; }

        public Vector<double> biases;
        public Matrix<double> weights;
        public Vector<double> in_vals;
        public Vector<double> ayes { get; set; }
        public Vector<double> deriv_ayes { get; set; }


        public ActivationFunction activationFunction { get; set; }

        public Layer(int inputs, int outputs)
        {
            this.inputs = inputs;
            this.outputs = outputs;

            this.biases = Vector<double>.Build.Dense(outputs);
            this.weights = Matrix<double>.Build.Dense(outputs, inputs);

        }

        public void resetBiases(IRandomVariable rv)
        {
            for (int j = 0; j < outputs; j++)
                biases[j] = rv.next();
        }

        public void resetWeights(IRandomVariable rv)
        {
            for (int i = 0; i < inputs; i++)
                for (int j = 0; j < outputs; j++)
                    weights[j, i] = rv.next();
        }

        public void activate(Vector<double> prevAyes)
        {
            in_vals = prevAyes;
            if (prevAyes.Count != inputs)
                throw new ArgumentOutOfRangeException("Expecting vector of size " + inputs + " but got " + prevAyes.Count);

            var awmid = weights.Multiply(prevAyes.ToColumnMatrix());
            var awplusb = awmid + biases.ToColumnMatrix();

            Matrix<double> ayesM = Matrix<double>.Build.Dense(outputs, 1);
            Matrix<double> deriv_ayesM = Matrix<double>.Build.Dense(outputs, 1);

            awplusb.Map(activationFunction.activate, ayesM);
            awplusb.Map(activationFunction.derivative, deriv_ayesM);

            ayes = ayesM.Column(0);
            deriv_ayes = deriv_ayesM.Column(0);
        }

        public double L2(Vector<double> expected)
        {
            if (ayes.Count != expected.Count)
                throw new ArgumentException("expected not same size as the data.");

            var v1 = ayes - expected;
            var v2 = v1.DotProduct(v1);
            return v2;
        }

        //
        // Here upvals is effectivel dC/d(a)L-1 where L-1 is the layer below.
        // For the bottom layer, some outside entity needs to pass in the correct values
        // calculated from the ayes.
        public Vector<double> train(Vector<double> upvals, double training_rate)
        {
            var dC_db = upvals.PointwiseMultiply(deriv_ayes); //Utils.haddamardProduct(upvals, deriv_ayes);
            var dC_da = weights.Transpose().Multiply(dC_db);
            var dC_dw = dC_db.ToColumnMatrix().Multiply(in_vals.ToRowMatrix()); // OuterProduct(in_vals);

            biases = biases.Subtract(dC_db.Multiply(training_rate));
            weights = weights.Subtract(dC_dw.Multiply(training_rate));

            return dC_da;
        }
    }
}
