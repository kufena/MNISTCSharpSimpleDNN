using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace CNN
{
    class ConvolutionalLayer
    {

        Convolution[] features;

        public ConvolutionalLayer(int convolutions, int rows, int cols)
        {
            features = new Convolution[convolutions];
            for(int i = 0; i < convolutions; i++)
            {
                features[i] = new Convolution(rows, cols);
            }
        }

        public void train() { }

        public void activate(Matrix<double>[] ins)
        {
            for(int i = 0; i < features.Length; i++)
            {
                for(int j = 0; j < ins.Length; j++)
                {
                    features[i].apply(ins[j]);
                }
            }
        }

    }
}
