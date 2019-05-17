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

        ILayer[] layers;

        public DNN(int nlayers, int[] dims)
        {
            this.numLayers = nlayers + 1;
            this.dims = dims;

            this.layers = new ILayer[numLayers];
            for(int i = 0; i < nlayers; i++)
            {
                var myLayer = new Layer(dims[i], dims[i + 1]);
                myLayer.resetBiases(new RandomVariables.UniformRV(1031, -0.5, +0.5)); // FixedValueRV(0.5));
                myLayer.resetWeights(new RandomVariables.UniformRV(1032, 0, 1)); // FixedValueRV(0.5));
                myLayer.activationFunction = new Activations.RELUActivation();
                this.layers[i] = myLayer;
            }
            this.layers[nlayers] = new SoftMax();

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
