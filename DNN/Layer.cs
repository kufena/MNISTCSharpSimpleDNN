using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace DNN
{
    public class Layer
    {

        int inputs { get; set; }
        int outputs { get; set; }

        public Vector<double> biases;
        public Matrix<double> weights;

        public Vector<double> ayes { get; set; }

        public ActivationFunction activationFunction { get;  set; }

        public Layer(int inputs, int outputs)
        {
            this.inputs = inputs;
            this.outputs = outputs;

            this.biases = Vector<double>.Build.Dense(outputs);
            this.weights = Matrix<double>.Build.Dense(inputs,outputs);

        }

        public void resetBiases(IRandomVariable rv)
        {
            for (int j = 0; j < outputs; j++)
                biases[j] = rv.next();
        }

        public void resetWeights(IRandomVariable rv) { 
            for (int i = 0; i < inputs; i++)
                for (int j = 0; j < outputs; j++)
                    weights[i, j] = rv.next();
        }

        public void activate(Vector<double> prevAyes)
        {
            if (prevAyes.Count != inputs)
                throw new ArgumentOutOfRangeException("Expecting vector of size " + inputs + " but got " + prevAyes.Count);

            var aw = weights.Multiply(prevAyes);
            var awplusb = aw + biases;

            ayes = Vector<double>.Build.Dense(outputs);
            for (int i = 0; i < outputs; i++)
                ayes[i] = activationFunction.activate(awplusb[i]);
        }
        public double L2(Vector<double> expected)
        {
            if (ayes.Count != expected.Count)
                throw new ArgumentException("expected not same size as the data.");

            var v1 = ayes - expected;
            var v2 = v1.DotProduct(v1);
            return v2;
        }
        
    }
}
