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
    public class DNN
    {
        int numLayers { get; set; }
        int[] dims { get; set; }
        public Vector<double> activation { get { return layers[layers.Length - 1].ayes;  } }
        ILayer[] layers;

        public DNN(int nlayers, int[] dims)
        {
            this.numLayers = nlayers + 1;
            this.dims = dims;

            this.layers = new ILayer[numLayers];
            for(int i = 0; i < nlayers; i++)
            {
                var myLayer = new Layer(dims[i], dims[i + 1]);
                myLayer.resetBiases(new UniformRV(1031, -0.5, +0.5)); // FixedValueRV(0.5));
                myLayer.resetWeights(new UniformRV(1032, -0.5, 0.5)); // FixedValueRV(0.5));
                myLayer.activationFunction = new Activations.RELUActivation();
                this.layers[i] = myLayer;
            }
            this.layers[nlayers] = new SoftMax();

        }

        public DNN()
        {
            this.numLayers = 2;
            this.dims = new int[] { 2, 2, 2 };
            this.layers = new ILayer[numLayers];
            for(int i = 0; i < numLayers; i++)
            {
                var mylayer = new Layer(dims[i], dims[i+1]);
                mylayer.resetBiases(new FixedValueRV(0.5)); //RandomVariables.UniformRV(1031, 0, 1));
                mylayer.resetWeights(new FixedValueRV(0.5)); //UniformRV(1032, 0, 1));
                mylayer.activationFunction = new Activations.RELUActivation();
                this.layers[i] = mylayer;
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

        public double train(Vector<double> ins, Vector<double> expect, double learningrate)
        {
            activate(ins);
            double res = layers[numLayers - 1].L2(expect);
            var derivs = layers[numLayers - 1].ayes.Subtract(expect);

            for (int i = numLayers - 1; i >= 0; i--)
            {
                derivs = layers[i].train(derivs, learningrate);
            }
            return res;
        }
    }
}
