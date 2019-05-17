using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

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

            var pam = prevAyes.ToColumnMatrix();
            var awmid = weights.Multiply(pam);
            var aw = awmid.Column(0);
            var awplusb = aw + biases;

            ayes = Vector<double>.Build.Dense(outputs);
            deriv_ayes = Vector<double>.Build.Dense(outputs);

            for (int i = 0; i < outputs; i++)
            {
                ayes[i] = activationFunction.activate(awplusb[i]);
                deriv_ayes[i] = activationFunction.derivative(awplusb[i]);
            }
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
        public Vector<double> train(Vector<double> upvals)
        {
            var dC_db = Utils.haddamardProduct(upvals, deriv_ayes);
            var dC_da = weights.Transpose().Multiply(dC_db);
            var dC_dw = dC_db.OuterProduct(in_vals);

            biases = biases.Subtract(dC_db.Multiply(0.01));
            weights = weights.Subtract(dC_dw.Multiply(0.01));

            return dC_da;
        }
    }
}
