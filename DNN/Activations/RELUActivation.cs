using System;
using System.Collections.Generic;
using System.Text;

namespace DNN.Activations
{
    class RELUActivation : ActivationFunction
    {
        public double activate(double z)
        {
            if (z > 0) return z;
            else return 0;
        }

        // we're using an arbitrary value for the derivative at zero as ReLu has
        // no derivate at zero.
        public double derivative(double zprime)
        {
            if (zprime > 0) return 1;
            return 0;
        }
    }
}
