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

        public double derivative(double zprime)
        {
            if (zprime > 0) return 1;
            return 0;
        }
    }
}
