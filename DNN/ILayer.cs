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
using MathNet.Numerics.LinearAlgebra;
using Utilities.RandomVariables;

namespace DNN
{
    public interface ILayer
    {
        ActivationFunction activationFunction { get; set; }
        Vector<double> ayes { get; set; }
        //Vector<double> deriv_ayes { get; set; }

        void activate(Vector<double> prevAyes);
        double L2(Vector<double> expected);
        void resetBiases(IRandomVariable rv);
        void resetWeights(IRandomVariable rv);
        Vector<double> train(Vector<double> upvals, double training_rate);
    }
}