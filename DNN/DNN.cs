using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace DNN
{
    public class DNN
    {
        int numLayers { get; set; }
        int[] dims { get; set; }

        Layer[] layers;

        public DNN(int nlayers, int[] dims)
        {
            this.numLayers = nlayers;
            this.dims = dims;

            this.layers = new Layer[nlayers];
            for(int i = 0; i < nlayers; i++)
            {
                this.layers[i] = new Layer(dims[i], dims[i + 1]);
                this.layers[i].resetBiases(new RandomVariables.FixedValueRV(0.5));
                this.layers[i].resetWeights(new RandomVariables.FixedValueRV(0.5));
                this.layers[i].activationFunction = new Activations.SigmoidActivation();
            }

        }

        public void activate(Vector<double> ins)
        {
            layers[0].activate(ins);
            for(int i = 1; i < numLayers; i++)
            {
                layers[i].activate(layers[i - 1].ayes);
            }
        }

        public double train(Vector<double> ins, Vector<double> expect)
        {
            activate(ins);
            double res = layers[numLayers - 1].L2(expect);
            var derivs = layers[numLayers - 1].ayes.Subtract(expect);

            for (int i = numLayers - 1; i >= 0; i--)
            {
                derivs = layers[i].train(derivs);
            }
            return res;
        }
    }
}
